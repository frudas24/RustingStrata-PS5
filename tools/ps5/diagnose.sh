#!/usr/bin/env bash
set -u
printf '%s\n' '=== OS / CPU ==='; uname -a; lscpu
printf '\n=== RAM ===\n'; grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
for d in /sys/class/drm/card*/device; do
  [[ -r "$d/vendor" ]] || continue
  printf '\n=== %s ===\n' "$d"
  for f in vendor device mem_info_vram_total mem_info_vram_used mem_info_gtt_total mem_info_gtt_used; do
    [[ -r "$d/$f" ]] && printf '%s=%s\n' "$f" "$(cat "$d/$f")"
  done
done
printf '\n=== Vulkan ===\n'
if command -v vulkaninfo >/dev/null; then
  vulkaninfo --summary
  vulkaninfo 2>/dev/null | grep -E 'heap(Budget|Usage)|memoryHeap|memoryType|DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|shaderFloat64|shaderInt(8|16|64)|shaderFloat16|shaderIntegerDotProduct|integerDotProduct(8Bit|4x8BitPacked).*Accelerated|subgroupSize|deviceCoherentMemory|vulkanMemoryModel|bufferDeviceAddress|maxComputeSharedMemorySize|VK_EXT_memory_budget|VK_AMD_device_coherent_memory' || true
else echo 'vulkaninfo missing (install your distro vulkan-tools package).'; fi
