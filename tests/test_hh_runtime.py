"""Integration checks for the controlled timing path and its fixed protocol."""

import unittest
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from studies.hh_convergence.runtime_benchmark import worker, run_benchmark
from studies.hh_convergence.runtime_benchmark.analyze import summarize_records


class ControlledRuntimeTests(unittest.TestCase):
    def test_fresh_repetitions_preserve_fields_and_restore_mask(self):
        original = worker.solver.gate_mask
        case = worker.FieldCase(3, 2, 8, .01, 24)
        a, timing_a = worker.fresh_solve(case, .4)
        self.assertIs(worker.solver.gate_mask, original)
        b, timing_b = worker.fresh_solve(case, .4)
        self.assertIs(worker.solver.gate_mask, original)
        worker.np.testing.assert_array_equal(a.voltage, b.voltage)
        worker.np.testing.assert_array_equal(a.gates, b.gates)
        self.assertGreater(float(a.voltage[-1].max()), 0)
        for timing in (timing_a, timing_b):
            self.assertGreater(timing['solve_cpu_seconds'], 0)
            self.assertGreaterEqual(timing['solve_wall_seconds'], timing['loop_wall_seconds'])

    def test_exception_restores_original_mask(self):
        original = worker.solver.gate_mask
        with patch.object(worker.IMEXBDFSolver, 'run', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                worker.fresh_solve(worker.FieldCase(2, 1, 8, .01, 24), .4)
        self.assertIs(worker.solver.gate_mask, original)

    def test_approved_schedule_and_runtime_thread_controls(self):
        plan = json.loads((run_benchmark.HERE / 'benchmark_plan.json').read_text())
        self.assertEqual(len(run_benchmark.validate_plan(plan)), 19)
        for repeat in (1, 2, 3):
            self.assertEqual(len({x['case'] for x in plan['schedule'] if x['repeat'] == repeat}), 19)
        # Match production: set thread controls in a fresh worker before NumPy imports.
        # Other tests may already have loaded a multithreaded BLAS in this process.
        probe = subprocess.run([sys.executable, '-B', '-c',
            'import json; from studies.hh_convergence.runtime_benchmark.worker import environment; '
            'print(json.dumps(environment()))'], cwd=run_benchmark.PROJECT,
            capture_output=True, text=True, check=True)
        info = json.loads(probe.stdout)
        self.assertTrue(all(value == '1' for value in info['thread_environment'].values()))
        if worker.os.name == 'nt':
            self.assertEqual({row['package'] for row in info['queried_libraries']}, {'numpy','scipy'})
        self.assertTrue(all(row['threads'] == 1 for row in info['queried_libraries']))

    def test_summary_requires_complete_distinct_fresh_measurements(self):
        import copy
        plan = json.loads((run_benchmark.HERE / 'benchmark_plan.json').read_text())
        records = []
        for item in plan['schedule']:
            value = {1:15., 2:8., 3:10.}[item['repeat']]
            records.append(dict(case_id=item['case'],repeat=item['repeat'],identity='test',
                environment={'threads':1},status='completed',fresh_solve=True,
                timing=dict(solve_wall_seconds=value,solve_cpu_seconds=value-1,loop_wall_seconds=value-.1),
                errors=dict(voltage_error=.1,gate_error=.001),
                historical_field_max_differences=dict(voltage=0.,gates=0.),
                diagnostics=dict(gate_mask_steps=0,voltage_cutoff_steps=0)))
        summaries = summarize_records(plan, records)
        self.assertEqual(len(summaries),19)
        self.assertEqual(summaries[0]['median_solve_wall_seconds'],10.)
        self.assertEqual(summaries[0]['mean_solve_wall_seconds'],11.)
        self.assertEqual(summaries[0]['mean_solve_cpu_seconds'],10.)
        self.assertAlmostEqual(summaries[0]['wall_relative_span'],.7)
        with self.assertRaisesRegex(ValueError,'three distinct'):
            summarize_records(plan,records[:-1])
        duplicate = copy.deepcopy(records)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError,'three distinct'):
            summarize_records(plan,duplicate)
        changed = copy.deepcopy(records)
        changed[0]['environment']['threads']=2
        with self.assertRaisesRegex(ValueError,'settings changed'):
            summarize_records(plan,changed)


if __name__ == '__main__':
    unittest.main()
