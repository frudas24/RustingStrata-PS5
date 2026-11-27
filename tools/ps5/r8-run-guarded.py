#!/usr/bin/env python3
"""Run one PS5 task and collect process/device samples. Kill only that process group."""
import argparse
import csv
from pathlib import Path
import json
import os
import resource
import signal
import subprocess
import time

def proc_fields(path, keys):
    try:
        lines = path.read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return dict.fromkeys(keys,'')
    values = {}
    for line in lines:
        key, sep, value = line.partition(':')
        if sep and key in keys:
            fields=value.split()
            values[key]=fields[0] if fields else ''
    return {key:values.get(key,'') for key in keys}

def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ''

def stop(child):
    if child.poll() is None:
        try: os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        try:
            child.wait(timeout=4)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()

def memory_ok(value,minimum):
    try: return int(value)>=minimum
    except (TypeError,ValueError): return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--min-kib',type=int,default=1048576)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.timeout<=0 or args.min_kib<=0: parser.error('command, timeout and memory guard required')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    mem=Path('/proc/meminfo')
    if not memory_ok(proc_fields(mem,('MemAvailable',))['MemAvailable'],args.min_kib):
        raise SystemExit(75)
    device=Path('/sys/class/drm/card0/device')
    hwmon=next((device/'hwmon').glob('hwmon*'),device)
    fields=['elapsed_s','MemAvailable','SwapFree','VmHWM','VmRSS','read_bytes','write_bytes',
            'gpu_busy_pct','vram_bytes','gtt_bytes','temp_mc','gpu_freq_hz']
    start=time.monotonic()
    reason=''
    with args.out.with_suffix('.telemetry.csv').open('w',newline='') as data:
        writer=csv.DictWriter(data,fields)
        writer.writeheader()
        child=subprocess.Popen(command,start_new_session=True)
        proc=Path('/proc')/str(child.pid)
        try:
            while child.poll() is None:
                row=proc_fields(mem,('MemAvailable','SwapFree'))
                row.update(proc_fields(proc/'status',('VmHWM','VmRSS')))
                row.update(proc_fields(proc/'io',('read_bytes','write_bytes')))
                row.update(elapsed_s=round(time.monotonic()-start,6),
                    gpu_busy_pct=read(device/'gpu_busy_percent'),vram_bytes=read(device/'mem_info_vram_used'),
                    gtt_bytes=read(device/'mem_info_gtt_used'),temp_mc=read(hwmon/'temp1_input'),
                    gpu_freq_hz=read(hwmon/'freq1_input'))
                writer.writerow(row)
                data.flush()
                if not memory_ok(row['MemAvailable'],args.min_kib):
                    reason='memory guard'
                    break
                if row['elapsed_s']>=args.timeout:
                    reason='timeout'
                    break
                time.sleep(.25)
        finally:
            stop(child)
    use=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(command=command,wall_s=time.monotonic()-start,child_exit=child.returncode,guard_reason=reason,
        max_rss_kib=use.ru_maxrss,minor_faults=use.ru_minflt,major_faults=use.ru_majflt,
        voluntary_cs=use.ru_nvcsw,involuntary_cs=use.ru_nivcsw,user_s=use.ru_utime,sys_s=use.ru_stime)
    args.out.with_suffix('.usage.json').write_text(json.dumps(result,indent=2)+'\n')
    return 75 if reason else (128-child.returncode if child.returncode<0 else child.returncode)

if __name__=='__main__':
    raise SystemExit(main())
