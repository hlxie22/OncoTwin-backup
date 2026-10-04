import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dynamic_scan" / "ckpt0_access_audit.py"

spec = importlib.util.spec_from_file_location("ckpt0", SCRIPT)
ckpt0 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ckpt0)


class Checkpoint0AuditTests(unittest.TestCase):
    def test_dataset_guessing(self):
        self.assertEqual(
            ckpt0.guess_dataset(Path("/x/GENIE_20/data.tsv")),
            "GENIE",
        )
        self.assertEqual(
            ckpt0.guess_dataset(Path("/x/MSK-CHORD/radiology.csv")),
            "CHORD",
        )
        self.assertEqual(
            ckpt0.guess_dataset(Path("/x/BPC/MSK/imaging.tsv")),
            "BPC_MSK",
        )
        self.assertEqual(
            ckpt0.guess_dataset(Path("/x/BPC/DFCI/imaging.tsv")),
            "BPC_DFCI",
        )
        self.assertEqual(
            ckpt0.guess_dataset(Path("/x/BPC/Vanderbilt/imaging.tsv")),
            "BPC_VICC",
        )

    def test_candidate_column_classification(self):
        out = ckpt0.classify_columns(
            [
                "PATIENT_ID",
                "SAMPLE_ID",
                "REPORT_DATE",
                "SEQ_ASSAY_ID",
                "Hugo_Symbol",
                "Radiology_Response",
                "Institution_Name",
            ]
        )

        self.assertIn("PATIENT_ID", out["patient_id"])
        self.assertIn("SAMPLE_ID", out["sample_id"])
        self.assertIn("REPORT_DATE", out["timestamp"])
        self.assertIn("SEQ_ASSAY_ID", out["panel_or_assay"])
        self.assertIn("Hugo_Symbol", out["gene_or_variant"])
        self.assertIn("Radiology_Response", out["imaging_or_response"])
        self.assertIn("Institution_Name", out["institution"])

    def test_delimited_schema_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "MSK-CHORD_scans.tsv"
            p.write_text(
                "PATIENT_ID\tREPORT_DATE\tMODALITY\tRESPONSE\n"
                "p1\t2025-01-01\tCT\tstable\n"
                "p2\t2025-02-01\tPET\tprogression\n",
                encoding="utf-8",
            )

            result = ckpt0.inspect_delimited(p)
            self.assertEqual(result["column_count"], 4)
            self.assertIn(
                "PATIENT_ID",
                result["candidate_columns"]["patient_id"],
            )
            self.assertIn(
                "REPORT_DATE",
                result["candidate_columns"]["timestamp"],
            )
            self.assertGreaterEqual(
                len(result["candidate_columns"]["imaging_or_response"]),
                1,
            )

    def test_external_policy_is_locked(self):
        policy_path = (
            ROOT
            / "configs"
            / "dynamic_scan"
            / "data_policy.json"
        )

        policy = json.loads(
            policy_path.read_text(encoding="utf-8")
        )

        errors = ckpt0.validate_policy(policy)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
