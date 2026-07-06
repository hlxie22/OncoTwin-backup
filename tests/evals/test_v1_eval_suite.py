from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.common import EvalResult, EvalUnavailable, require_modules
from evals.prior_stack.run_v1_eval_suite import run_suite
from evals.prior_stack.v1_posterior_health_eval import run_eval as run_posterior_health


class V1EvalSuiteTest(unittest.TestCase):
    def test_suite_runs_without_cohort_or_optional_runtimes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "suite.md"
            results = run_suite(report_path=report)
            self.assertTrue(report.exists())

        self.assertTrue(any(r.name == "real_data_prior_layer_performance" for r in results))
        self.assertTrue(any(r.status == "unavailable" for r in results))
        self.assertTrue(all(isinstance(r, EvalResult) for r in results))

    def test_stub_reports_missing_runtime(self):
        result = run_posterior_health()
        self.assertEqual(result.status, "unavailable")
        self.assertIn(
            "experiments.prior_builder.bayesian_update",
            result.missing_components,
        )

    def test_require_modules_raises_clear_error(self):
        with self.assertRaises(EvalUnavailable) as raised:
            require_modules(
                "missing_eval",
                ("not_a_real_oncotwin_runtime",),
                "Missing runtime",
            )
        self.assertEqual(raised.exception.name, "missing_eval")
        self.assertEqual(
            raised.exception.missing,
            ("not_a_real_oncotwin_runtime",),
        )


if __name__ == "__main__":
    unittest.main()
