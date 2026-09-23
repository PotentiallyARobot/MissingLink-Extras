import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from h3_swaps import sam_checkpoint, public_sam_checkpoint


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

    def test_bucket_download_includes_license_and_reuses_valid_weights(self):
        result = Path(sam_checkpoint())
        self.assertEqual(result.read_bytes(), b"test weights")
        self.assertTrue(result.with_name("LICENSE").exists())
        self.hub.HfApi.assert_called_with()
        self.api.download_bucket_files.reset_mock()
        sam_checkpoint()
        self.assertEqual(self.api.download_bucket_files.call_count, 1)

    def test_corrupt_download_is_never_promoted(self):
        self.data["sam3.pt"] = b"corrupt"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            sam_checkpoint()
        self.assertFalse((Path(self.temp.name) / self.revision / "sam3.pt").exists())

    def test_inaccessible_bucket_uses_ungated_public_mirror(self):
        self.api.download_bucket_files.side_effect = FileNotFoundError()
        with patch("h3_swaps.public_sam_checkpoint", return_value="public.pt") as fallback:
            self.assertEqual(sam_checkpoint(), "public.pt")
            fallback.assert_called_once_with(Path(self.temp.name))
        self.hub.hf_hub_download.assert_not_called()

    def test_manifest_cannot_escape_cache_directory(self):
        self.manifest["revision"] = "../escape"
        with self.assertRaisesRegex(ValueError, "Invalid SAM bucket revision"):
            sam_checkpoint()

    def test_public_fallback_uses_pinned_anonymous_download_and_atomic_conversion(self):
        safe = types.SimpleNamespace(load_file=Mock(return_value={"detector.weight": "tensor"}))
        torch = types.SimpleNamespace(save=Mock(side_effect=lambda weights, path: Path(path).write_bytes(b"converted")))
        with patch.dict(sys.modules, {"safetensors.torch": safe, "torch": torch}):
            result = Path(public_sam_checkpoint(Path(self.temp.name)))
            self.assertEqual(result.read_bytes(), b"converted")
            public_sam_checkpoint(Path(self.temp.name))
        torch.save.assert_called_once()
        for call in self.hub.hf_hub_download.call_args_list:
            self.assertEqual(call.args[0], "AEmotionStudio/sam3")
            self.assertIs(call.kwargs["token"], False)
            self.assertEqual(len(call.kwargs["revision"]), 40)

    def test_public_fallback_rejects_incompatible_weights(self):
        safe = types.SimpleNamespace(load_file=Mock(return_value={"detector_model.weight": "wrong format"}))
        with patch.dict(sys.modules, {"safetensors.torch": safe, "torch": Mock()}):
            with self.assertRaisesRegex(ValueError, "checkpoint keys"):
                public_sam_checkpoint(Path(self.temp.name))


if __name__ == "__main__":
    unittest.main()
