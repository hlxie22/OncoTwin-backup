import json
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "schemas" / "mechanistic_simulator"
FIXTURE_ROOT = REPO_ROOT / "fixtures" / "mechanistic_simulator"


def load_json(path):
    with path.open() as f:
        return json.load(f)


@unittest.skipIf(jsonschema is None, "jsonschema is not installed")
class MechanisticSchemaValidationTests(unittest.TestCase):
    def validate_file(self, fixture_path, schema_path):
        schema = load_json(schema_path)
        instance = load_json(fixture_path)
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(instance)

    def test_case_fixtures_validate(self):
        schema_path = SCHEMA_ROOT / "case_fixture.schema.json"
        for fixture_path in sorted((FIXTURE_ROOT / "cases").glob("*.json")):
            with self.subTest(fixture=str(fixture_path.relative_to(REPO_ROOT))):
                self.validate_file(fixture_path, schema_path)

    def test_treatment_schedules_validate(self):
        schema_path = SCHEMA_ROOT / "treatment_schedule.schema.json"
        for fixture_path in sorted((FIXTURE_ROOT / "schedules").glob("*.json")):
            with self.subTest(fixture=str(fixture_path.relative_to(REPO_ROOT))):
                self.validate_file(fixture_path, schema_path)

    def test_parameter_fixtures_validate(self):
        for fixture_path in sorted((FIXTURE_ROOT / "params").glob("*.json")):
            if fixture_path.name == "generic_volume_prior.json":
                schema_path = SCHEMA_ROOT / "prior_config.schema.json"
            else:
                schema_path = SCHEMA_ROOT / "resolved_params.schema.json"

            with self.subTest(
                fixture=str(fixture_path.relative_to(REPO_ROOT)),
                schema=str(schema_path.relative_to(REPO_ROOT)),
            ):
                self.validate_file(fixture_path, schema_path)


if __name__ == "__main__":
    unittest.main()
