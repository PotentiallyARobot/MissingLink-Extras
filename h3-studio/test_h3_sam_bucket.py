import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from h3_swaps import sam_checkpoint


class BucketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.api = Mock()
        self.hub = types.SimpleNamespace(HfApi=Mock(return_value=self.api), hf_hub_download=Mock(return_value="official.pt"))
        self.enterContext(patch.dict(sys.modules, huggingface_hub=self.hub))
        self.enterContext(patch.dict(os.environ, H3_SAM_CACHE=self.temp.name))
        self.revision = "a" * 40
        self.data = {"LICENSE": b"license", "config.json": b"{}", "sam3.pt": b"test weights"}
        self.manifest = {"source": "facebook/sam3", "revision": self.revision,
                         "prefix": "models/sam3/" + self.revision,
                         "files": {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                                   for n, b in self.data.items()}}
        def download(bucket, files):
            for remote, local in files:
                content = json.dumps(self.manifest).encode() if remote.endswith("manifest.json") else self.data[remote.rsplit("/", 1)[-1]]
                Path(local).write_bytes(content)
        self.api.download_bucket_files.side_effect = download

    def test_anonymous_download_includes_license_and_reuses_valid_weights(self):
        result = Path(sam_checkpoint())
        self.assertEqual(result.read_bytes(), b"test weights")
        self.assertTrue(result.with_name("LICENSE").exists())
        self.hub.HfApi.assert_called_with(token=False)
        self.api.download_bucket_files.reset_mock()
        sam_checkpoint()
        self.assertEqual(self.api.download_bucket_files.call_count, 1)

    def test_corrupt_download_is_never_promoted(self):
        self.data["sam3.pt"] = b"corrupt"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            sam_checkpoint()
        self.assertFalse((Path(self.temp.name) / self.revision / "sam3.pt").exists())

    def test_unpublished_mirror_uses_authorized_official_download(self):
        self.api.download_bucket_files.side_effect = FileNotFoundError()
        self.assertEqual(sam_checkpoint(), "official.pt")
        self.hub.hf_hub_download.assert_called_once_with("facebook/sam3", "sam3.pt")

    def test_manifest_cannot_escape_cache_directory(self):
        self.manifest["revision"] = "../escape"
        with self.assertRaisesRegex(ValueError, "Invalid SAM bucket revision"):
            sam_checkpoint()


if __name__ == "__main__":
    unittest.main()
