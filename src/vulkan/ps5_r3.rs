//! Experimental immutable file-backed host import. No writes, no expert slots,
//! no converted tensors. Driver rejection disables subsequent attempts.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
const HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw());

pub(super) struct Config { align: usize, cap: u64, max_alloc: u64 }
pub(super) fn probe(i: &ash::Instance, pd: vk::PhysicalDevice, allowed: bool, ext: bool) -> anyhow::Result<Option<Config>> {
    if !std::env::var("STRATA_PS5_ZERO_COPY").is_ok_and(|v| v == "1") { return Ok(None); }
    if !cfg!(target_os = "linux") || !allowed || !ext {
        crate::say!("PS5 R3: host import unavailable (requires Linux AMD PS5 GTT/budget and extension); R1 upload");
        return Ok(None);
    }
    let mut p = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut m = vk::PhysicalDeviceMaintenance3Properties::default();
    unsafe { i.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut p).push_next(&mut m)) };
    let align = usize::try_from(p.min_imported_host_pointer_alignment)?;
    let mut props = vk::ExternalBufferProperties::default();
    unsafe { i.get_physical_device_external_buffer_properties(pd, &vk::PhysicalDeviceExternalBufferInfo::default().usage(USAGE).handle_type(HANDLE), &mut props) };
    let e = props.external_memory_properties;
    if !align.is_power_of_two() || align < 16 || !e.compatible_handle_types.contains(HANDLE)
        || !e.external_memory_features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        || e.external_memory_features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY) {
        crate::say!("PS5 R3: incompatible host import requirements; R1 upload");
        return Ok(None);
    }
    let mb = std::env::var("STRATA_PS5_ZERO_COPY_MB").unwrap_or_else(|_| "512".into()).parse::<u64>().context("STRATA_PS5_ZERO_COPY_MB")?;
    ensure!(mb <= 8192, "STRATA_PS5_ZERO_COPY_MB must be <= 8192");
    if mb == 0 { return Ok(None); }
    crate::say!("PS5 R3: host import experimental, alignment {align}, cap {mb} MiB");
    Ok(Some(Config { align, cap: mb << 20, max_alloc: m.max_memory_allocation_size }))
}

pub(super) struct HostImport {
    cfg: Config, ext: ash::ext::external_memory_host::Device,
    disabled: AtomicBool, live: AtomicU64, imported: AtomicU64, attempts: AtomicU64, rejected: AtomicU64,
    // Serialize admission, driver import and registration across shared backend handles.
    gate: Mutex<()>,
}
impl HostImport {
    pub(super) fn new(cfg: Config, i: &ash::Instance, d: &ash::Device) -> Self {
        Self { cfg, ext: ash::ext::external_memory_host::Device::new(i, d), disabled: false.into(), live: 0.into(), imported: 0.into(), attempts: 0.into(), rejected: 0.into(), gate: Mutex::new(()) }
    }
    pub(super) fn release(&self, bytes: u64) { self.live.fetch_sub(bytes, Relaxed); }
    pub(super) fn report(&self) {
        eprintln!("PS5 R3: imported_bytes={} attempts={} rejected={} live_span_bytes={}", self.imported.load(Relaxed), self.attempts.load(Relaxed), self.rejected.load(Relaxed), self.live.load(Relaxed));
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Window { start: usize, len: usize, offset: usize }
/// Never round outside the mapped file, including its last partial page.
fn window(base: usize, file_len: usize, offset: usize, len: usize, align: usize) -> Option<Window> {
    if !align.is_power_of_two() || align < 16 || len < 65536 || len % 4 != 0 { return None; }
    let p = base.checked_add(offset)?;
    if p % 16 != 0 { return None; }
    let end = p.checked_add(len)?;
    let a = p & !(align - 1);
    let z = end.checked_add(align - 1)? & !(align - 1);
    if a < base || z > base.checked_add(file_len)? { return None; }
    Some(Window { start: a - base, len: z - a, offset: p - a })
}

impl Vk {
    pub(super) fn try_import_weight(&self, t: &crate::weights::Tensor, source: &crate::gguf::Gguf) -> anyhow::Result<Option<Buf>> {
        let Some(r) = &self.0.r3 else { return Ok(None); };
        // Restrict initial experiment to familiar quantized read-only weight formats.
        if t.rows.checked_mul(t.ty.row_bytes(t.cols as u64)) != Some(t.data.len()) { return Ok(None); }
        if !matches!(t.ty, GgmlType::Q4K | GgmlType::Q8_0 | GgmlType::Q2K | GgmlType::IQ4XS | GgmlType::IQ3XXS) { return Ok(None); }
        let Some((map, offset)) = source.mapped_source(t.data) else { return Ok(None); };
        let Some(w) = window(map.0.as_ptr() as usize, map.0.len(), offset, t.data.len(), r.cfg.align) else { return Ok(None); };
        let n = w.len as u64;
        let _gate = r.gate.lock().unwrap_or_else(|e| e.into_inner());
        if r.disabled.load(Relaxed) || n > r.cfg.max_alloc || n > r.cfg.cap.saturating_sub(r.live.load(Relaxed)) || n > self.0.ps5_host_room() { return Ok(None); }
        // Source is an immutable mmap. Some Linux AMD userptr paths refuse it;
        // never turn it into a writable COW map or copy it behind the user's back.
        let ptr = unsafe { map.0.as_ptr().add(w.start) }.cast::<std::ffi::c_void>();
        let protected = crate::weights::ImportedRange::new(ptr as usize, w.len);
        r.attempts.fetch_add(1, Relaxed);
        let res = self.import_window(r, ptr, &w);
        let (buf, mem, addr) = match res {
            Ok(v) => v,
            Err(e) => {
                if e.downcast_ref::<vk::Result>() == Some(&vk::Result::ERROR_DEVICE_LOST) { return Err(e); }
                r.rejected.fetch_add(1, Relaxed); r.disabled.store(true, Relaxed);
                crate::say!("PS5 R3: host import rejected ({e:#}); disabling imports, R1 upload");
                return Ok(None);
            }
        };
        // Never replace an existing BDA registration (including partially overlapping ranges).
        {
            let mut st = self.0.st();
            let prev = st.bufs.range(..=addr).next_back().is_some_and(|(&a, &(_, len))| a.saturating_add(len as u64) > addr);
            let next = st.bufs.range(addr..).next().is_some_and(|(&a, _)| a < addr + n);
            if prev || next {
                unsafe { self.0.dev.destroy_buffer(buf, None); self.0.dev.free_memory(mem, None); }
                r.rejected.fetch_add(1, Relaxed); r.disabled.store(true, Relaxed);
                crate::say!("PS5 R3: overlapping device address; disabling imports, R1 upload");
                return Ok(None);
            }
            st.bufs.insert(addr, (buf, w.len));
        }
        r.live.fetch_add(n, Relaxed); r.imported.fetch_add(t.data.len() as u64, Relaxed);
        let owner = Mem { vk: self.0.clone(), buf, mem, addr, imported: Some((map, n, protected)) };
        Ok(Some(Buf::new(addr + w.offset as u64, t.data.len(), owner).read_only()))
    }

    fn import_window(&self, r: &HostImport, ptr: *const std::ffi::c_void, w: &Window) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory, u64)> {
        let d = &self.0.dev;
        let mut hp = vk::MemoryHostPointerPropertiesEXT::default();
        let result = unsafe { (r.ext.fp().get_memory_host_pointer_properties_ext)(d.handle(), HANDLE, ptr, &mut hp) };
        if result != vk::Result::SUCCESS { return Err(result.into()); }
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE);
        let mut bi = vk::BufferCreateInfo::default().size(w.len as u64).usage(USAGE).push_next(&mut external);
        if self.0.tq.is_some() { bi = bi.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&self.0.qfs); }
        let buf = unsafe { d.create_buffer(&bi, None) }?;
        let result = (|| -> anyhow::Result<(vk::DeviceMemory, u64)> {
            let req = unsafe { d.get_buffer_memory_requirements(buf) };
            ensure!(req.size <= w.len as u64, "driver buffer padding exceeds mapped window");
            let bits = req.memory_type_bits & hp.memory_type_bits;
            let budget = self.0.heap_budget();
            let idx = (0..self.0.mem.memory_type_count).find(|&i| {
                let t = self.0.mem.memory_types[i as usize];
                bits & (1 << i) != 0 && t.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED)
                    && (t.property_flags & vk::MemoryPropertyFlags::from_raw(0xc0)).is_empty()
                    && self.0.heap_free(&budget, t.heap_index as usize) >= w.len as u64
            }).context("no coherent cached host memory type within heap budget")?;
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default().handle_type(HANDLE).host_pointer(ptr.cast_mut());
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(w.len as u64).memory_type_index(idx).push_next(&mut import).push_next(&mut flags);
            let mem = unsafe { d.allocate_memory(&ai, None) }?;
            if let Err(e) = unsafe { d.bind_buffer_memory(buf, mem, 0) } {
                unsafe { d.free_memory(mem, None) }; return Err(e.into());
            }
            let addr = unsafe { d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf)) };
            if addr == 0 || addr.checked_add(w.len as u64).is_none() || (addr + w.offset as u64) % 16 != 0 {
                // Destroy the bound buffer before freeing its memory, in this error path too.
                unsafe { d.destroy_buffer(buf, None); d.free_memory(mem, None); }
                return Ok((vk::DeviceMemory::null(), 0));
            }
            Ok((mem, addr))
        })();
        match result {
            Ok((_, 0)) => anyhow::bail!("invalid imported buffer device address"),
            Ok((mem, addr)) => Ok((buf, mem, addr)),
            Err(e) => { unsafe { d.destroy_buffer(buf, None) }; Err(e) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn windows() {
        assert_eq!(window(0x10000, 0x40000, 32, 65536, 4096), Some(Window {start: 0, len: 69632, offset: 32}));
        assert!(window(0x10000, 65568, 32, 65536, 4096).is_none());
        assert!(window(0x10000, 0x40000, 1, 65536, 4096).is_none());
        assert!(window(0x10000, 0x40000, 32, 65535, 4096).is_none());
        assert!(window(usize::MAX - 15, 0x40000, 32, 65536, 4096).is_none());
        assert!(window(0x10020, 0x40000, 0, 65536, 4096).is_none());
    }

    #[test]
    #[ignore = "requires real AMD Vulkan device; run tools/ps5/test-r3.sh"]
    fn gpu_import_parity() {
        use crate::{gguf::Gguf, weights::Tensor};
        let gpu = Vk::new(0).expect("Vulkan required");
        let cols = 256usize;
        let rows = 512usize;
        let ty = GgmlType::Q4K;
        let n = rows * ty.row_bytes(cols as u64);
        let raw: Vec<u8> = include_bytes!("../../tests/fixtures/Q4_K.raw").iter().copied().cycle().take(n).collect();
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes()); bytes.extend(1u64.to_le_bytes()); bytes.extend(1u64.to_le_bytes());
        let key = b"general.alignment";
        bytes.extend((key.len() as u64).to_le_bytes()); bytes.extend_from_slice(key);
        bytes.extend(4u32.to_le_bytes()); bytes.extend(65536u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes()); bytes.push(b'w'); bytes.extend(2u32.to_le_bytes());
        bytes.extend((cols as u64).to_le_bytes()); bytes.extend((rows as u64).to_le_bytes());
        bytes.extend(12u32.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(65536), 0);
        bytes.extend_from_slice(&raw); bytes.resize(bytes.len().next_multiple_of(65536) + 65536, 0);
        let p = std::env::temp_dir().join(format!("strata-r3-{}-{}.gguf", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        { use std::io::Write; let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&p).unwrap(); f.write_all(&bytes).unwrap(); }
        let source = Gguf::open(&p).unwrap();
        let data = source.bytes(&source.tensors["w"]);
        let (map, _) = source.mapped_source(data).unwrap(); let weak = Arc::downgrade(&map); drop(map);
        assert!(source.mapped_source(&raw).is_none());
        let t = Tensor { ty, rows, cols, data, scale: 1.0 };
        let copied = gpu.upload(&t).unwrap();
        let candidate = gpu.upload_immutable(&t, &source).unwrap();
        let imported = candidate.is_read_only();
        eprintln!("PS5 R3 test: imported={imported}");
        if std::env::var("PS5_R3_REQUIRE_IMPORT").is_ok_and(|v| v == "1") { assert!(imported, "driver did not import file pages"); }
        if std::env::var("STRATA_PS5_ZERO_COPY").is_ok_and(|v| v == "0") { assert!(!imported); }
        // The mapping must outlive Gguf; no borrowing tensor survives here.
        drop(source);
        assert_eq!(weak.upgrade().is_some(), imported);
        let be: &dyn Backend = &gpu;
        // Exercise public write guards even on devices that reject host imports.
        let ro = gpu.alloc(16).unwrap().read_only();
        let rw = gpu.alloc(16).unwrap();
        assert!(gpu.htod(&[0; 4], &ro, 0).is_err());
        assert!(gpu.dtod(&rw, 0, &ro, 0, 4).is_err());
        assert!(gpu.memset(&ro).is_err());
        let x = be.zeros::<f32>(cols).unwrap();
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 - 128.0) / 127.0).collect();
        be.put(&input, &x, 0).unwrap();
        let a = be.zeros::<f32>(rows).unwrap(); let b = be.zeros::<f32>(rows).unwrap();
        gpu.gemv(&copied, x.addr(), a.addr()).unwrap(); gpu.gemv(&candidate, x.addr(), b.addr()).unwrap();
        let a: Vec<f32> = be.get(&a, 0, rows).unwrap(); let b: Vec<f32> = be.get(&b, 0, rows).unwrap();
        assert!(a.iter().all(|x| x.is_finite()) && a.iter().any(|x| *x != 0.0));
        assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        drop(candidate); assert!(weak.upgrade().is_none());
        assert_eq!(std::fs::read(&p).unwrap(), bytes); std::fs::remove_file(p).unwrap();
    }
}
