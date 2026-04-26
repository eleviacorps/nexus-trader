import json
import unittest

from config.project_config import DATASET_MANIFEST_PATH


class DatasetManifestTests(unittest.TestCase):
    def test_manifest_has_expected_categories(self):
        manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertIn("macro", manifest)
        self.assertIn("news", manifest)
        self.assertIn("crowd", manifest)
        self.assertTrue(manifest["macro"])
        self.assertTrue(manifest["news"])
        self.assertTrue(manifest["crowd"])


if __name__ == "__main__":
    unittest.main()
