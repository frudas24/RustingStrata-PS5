#!/usr/bin/env python3
"""Offline regression models and tests of the real quality-gate parser (not GPU execution)."""
import collections
import importlib.util
import pathlib
import random
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('quality', pathlib.Path(__file__).with_name('quality-r11.py'))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def schedule(ids, capacity):
    # Independent model of the shader's workgroup policy. Atomics can order slots arbitrarily.
    for k, expert in enumerate(ids):
        if expert < 0:
            continue
        matches = [i for i, value in enumerate(ids) if value == expert]
        if capacity > 1 and 8 < len(ids) <= 256 and len(matches) <= capacity:
            if k == min(matches):
                yield from reversed(matches)
        else:
            yield k


class Audit(unittest.TestCase):
    def test_dedup_coverage(self):
        rng = random.Random(137)
        cases = 0
        for _ in range(5000):
            n = rng.choice([1, 8, 9, 16, 17, 18, 32, 128, 256, 257, rng.randrange(1, 260)])
            ids = [rng.randrange(-1, rng.choice([1, 2, 8, 32])) for _ in range(n)]
            want = collections.Counter(i for i, e in enumerate(ids) if e >= 0)
            for cap in (1, 4, 8, 17):
                self.assertEqual(collections.Counter(schedule(ids, cap)), want)
                cases += 1
        print(f'{cases} dedup coverage scenarios PASS (independent model)')

    def test_old_overflow_counterexample(self):
        ids = [7] * 18
        self.assertEqual(list(schedule(ids, 17)), list(range(18)))
        # Old policy wrote at most 17 matching slots and suppressed subsequent groups.
        self.assertNotEqual(set(range(17)), set(range(18)))

    def test_quality_dump(self):
        n, nv, nc = 4, 3, 2
        header = b'_logits_' + struct.pack('<III', n, nv, nc)
        tokens = struct.pack('<8I', *([0, 1, 2, 0] * 2))
        row = struct.pack('<ff4H', 0.001, -3.0, 0, 7, 12, 0)
        valid = header + tokens + row * 2
        log = 'PPL = 2.0 over 2 chunks of 4 tokens (2 scored)\n'
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / 'dump.bin'
            p.write_bytes(valid)
            self.assertEqual(quality.check_dump(p, log, n, nc), (n, nv, nc))
            cases = [
                (valid[:15], log), (valid[:-1], log), (valid+b'\0', log),
                (header+struct.pack('<I', 3)+tokens[4:]+row*2, log),
                (header+tokens+struct.pack('<f', float('nan'))+row[4:]+row, log),
                (header+tokens+struct.pack('<f', -1.0)+row[4:]+row, log),
                (header+tokens+row[:4]+struct.pack('<f', float('-inf'))+row[8:]+row, log),
                (valid, log.replace('2 scored', '1 scored')),
                (valid, log.replace('2.0', 'nan')), (valid, log+log), (valid, ''),
            ]
            for data, text in cases:
                p.write_bytes(data)
                with self.assertRaises(quality.r6.Reject):
                    quality.check_dump(p, text, n, nc)
            print(f'{len(cases)} invalid quality evidence fixtures rejected')


if __name__ == '__main__':
    unittest.main(verbosity=2)
