//! Opt-in PS5 prefill scheduling. Resident experts never move; only cache misses may be split.
//! All timing is local to one model instance, not saved across devices or models.
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode { Auto, Gpu, Half, Cpu }
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Key(pub usize, pub usize, pub usize, pub usize, pub usize);
fn bucket(x: usize) -> usize { if x == 0 { 0 } else { x.checked_next_power_of_two().unwrap_or(usize::MAX) } }
impl Key {
    pub fn new(layer: usize, tokens: usize, resident_rows: usize, missed_rows: usize, eligible: usize) -> Self {
        Self(layer, bucket(tokens), bucket(resident_rows), bucket(missed_rows), bucket(eligible))
    }
}
#[derive(Clone, Copy, Default)]
struct Sample { n: u64, ns: f64 }
#[derive(Default)]
struct History { arms: [Sample; 3], visits: u64 }
impl History {
    fn choose(&mut self, arms: usize) -> usize {
        let visit = self.visits;
        self.visits = self.visits.saturating_add(1);
        for a in 0..arms { if self.arms[a].n == 0 { return a; } }
        // Periodic deterministic exploration; no random source and no forced movement of resident experts.
        if visit % 16 == 15 { return ((visit / 16) % arms as u64) as usize; }
        let mut best = 0;
        for a in 1..arms {
            if self.arms[a].ns < self.arms[best].ns * 0.90 { best = a; }
        }
        best
    }
    fn observe(&mut self, arm: usize, ns: u64) {
        let s = &mut self.arms[arm];
        let v = ns.max(1) as f64;
        s.ns = if s.n == 0 { v } else { 0.8 * s.ns + 0.2 * v };
        s.n = s.n.saturating_add(1);
    }
}

pub(crate) struct Policy {
    mode: Mode,
    pub max_rows: usize,
    pub scratch_bytes: usize,
    reserve: u64,
    history: BTreeMap<Key, History>,
    pub stats: [u64; 3], // trials, CPU rows, trials with both CPU and GPU rows
}
impl Policy {
    pub fn from_env(ps5_preset: bool) -> anyhow::Result<Option<Self>> {
        if !ps5_preset || !cfg!(target_os = "linux") { return Ok(None); }
        let mode = match std::env::var("STRATA_PS5_SCHED").as_deref().unwrap_or("off") {
            "off" => return Ok(None), "auto" => Mode::Auto, "gpu" => Mode::Gpu, "half" => Mode::Half, "cpu" => Mode::Cpu,
            x => anyhow::bail!("STRATA_PS5_SCHED must be off, auto, gpu, half or cpu; got {x}"),
        };
        // R6 physical 35B gate: CPU-bearing arms changed logits and argmax. GPU-only is safe to measure.
        anyhow::ensure!(mode == Mode::Gpu || std::env::var("STRATA_PS5_R4_RESEARCH").is_ok_and(|v| v == "1"),
            "PS5 R7: hybrid prefill failed the physical R4 parity gate; use off/gpu (research only: STRATA_PS5_R4_RESEARCH=1)");
        fn number(name: &str, default: usize, max: usize) -> anyhow::Result<usize> {
            let n = match std::env::var(name) { Ok(s) => s.parse::<usize>()?, Err(std::env::VarError::NotPresent) => default, Err(e) => return Err(e.into()) };
            anyhow::ensure!(n > 0 && n <= max, "{name} must be in 1..={max}"); Ok(n)
        }
        let p = Self { mode, max_rows: number("STRATA_PS5_SCHED_MAX_ROWS", 4, 32)?,
            scratch_bytes: number("STRATA_PS5_SCHED_SCRATCH_MB", 64, 512)? << 20,
            reserve: (number("STRATA_PS5_HOST_RESERVE_MB", 1536, 65536)? as u64) << 20,
            history: BTreeMap::new(), stats: [0; 3] };
        crate::say!("PS5 R4: mode={mode:?} max_rows={} scratch_bytes={}", p.max_rows, p.scratch_bytes);
        Ok(Some(p))
    }
    pub fn budget(&self) -> usize {
        // Fail closed if the OS does not expose MemAvailable. This is a planning bound, not an allocation guarantee.
        let room = crate::memory::available_ram().unwrap_or(0).saturating_sub(self.reserve);
        self.scratch_bytes.min(usize::try_from(room).unwrap_or(usize::MAX))
    }
    pub fn choose(&mut self, key: Key, eligible: usize) -> usize {
        if eligible == 0 { return 0; }
        // Arms 0 GPU, 1 half, 2 all eligible. With one expert, half and all coincide.
        match self.mode {
            Mode::Gpu => 0, Mode::Half => 1, Mode::Cpu => if eligible == 1 { 1 } else { 2 },
            Mode::Auto => {
                if self.history.len() >= 4096 && !self.history.contains_key(&key) { return 0; }
                self.history.entry(key).or_default().choose(if eligible == 1 { 2 } else { 3 })
            }
        }
    }
    pub fn observe(&mut self, key: Key, arm: usize, ns: u64, cpu_rows: usize, gpu_rows: usize) {
        self.stats[0] = self.stats[0].saturating_add(1);
        self.stats[1] = self.stats[1].saturating_add(cpu_rows as u64);
        self.stats[2] = self.stats[2].saturating_add((cpu_rows > 0 && gpu_rows > 0) as u64);
        if let Some(h) = self.history.get_mut(&key) { h.observe(arm, ns); }
    }
}

/// Select only whole experts with few routed rows; keep original order for CPU scatter and GPU staging.
/// The scratch estimate includes input readback, intermediate/quantized activations and per-pair bookkeeping.
/// It excludes OS page cache and allocator/Rayon overhead; the margin and opt-in gate remain necessary.
pub(crate) fn candidates(missed: &[usize], off: &[usize], tokens: usize, ne: usize, ff: usize, max_rows: usize, budget: usize) -> Vec<usize> {
    let Some(input) = tokens.checked_mul(ne).and_then(|x| x.checked_mul(8)) else { return vec![]; };
    let Some(pair) = ne.checked_add(ff).and_then(|x| x.checked_mul(64)).and_then(|x| x.checked_add(512)) else { return vec![]; };
    if budget <= input { return vec![]; }
    let mut left = (budget - input) / pair;
    let mut ordered: Vec<_> = missed.iter().copied().filter_map(|e| {
        let n = off.get(e.checked_add(1)?)?.checked_sub(*off.get(e)?)?;
        (n > 0 && n <= max_rows).then_some((n, e))
    }).collect();
    ordered.sort_unstable();
    ordered.into_iter().take_while(|&(n, _)| {
        if n > left { false } else { left -= n; true }
    }).map(|(_, e)| e).collect()
}

pub(crate) fn split(missed: &mut Vec<usize>, eligible: &[usize], arm: usize) -> Vec<usize> {
    let take = match arm { 1 => eligible.len().div_ceil(2), 2 => eligible.len(), _ => 0 };
    let mut cpu = eligible[..take].to_vec();
    cpu.sort_unstable();
    missed.retain(|e| cpu.binary_search(e).is_err());
    cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn learns_and_explores() {
        let mut h = History::default();
        for a in 0..3 { assert_eq!(h.choose(3), a); h.observe(a, [100, 50, 70][a]); }
        for _ in 3..15 { assert_eq!(h.choose(3), 1); h.observe(1, 50); }
        assert_eq!(h.choose(3), 0);
        h.observe(0, 10); // one outlier cannot immediately overturn the EWMA
        assert_eq!(h.choose(3), 1);
    }
    #[test] fn memory_and_partition() {
        let off = [0, 1, 5, 7, 17]; let original = vec![0, 1, 2, 3];
        let pair = 64 * (32 + 64) + 512; let input = 17 * 32 * 8;
        assert!(candidates(&original, &off, 17, 32, 64, 4, input).is_empty());
        let e = candidates(&original, &off, 17, 32, 64, 4, input + 3 * pair);
        assert_eq!(e, [0, 2]);
        for arm in 0..3 {
            let mut gpu = original.clone(); let cpu = split(&mut gpu, &e, arm);
            let mut both = gpu.clone(); both.extend(&cpu); both.sort_unstable();
            assert_eq!(both, original); assert!(cpu.iter().all(|i| !gpu.contains(i)));
        }
        assert!(candidates(&original, &off, usize::MAX, 32, 64, 4, usize::MAX).is_empty());
    }
    #[test] fn one_candidate_has_two_arms() {
        let mut h = History::default();
        assert_eq!(h.choose(2), 0); h.observe(0, 100);
        assert_eq!(h.choose(2), 1); h.observe(1, 50);
        assert_eq!(h.choose(2), 1);
    }
}
