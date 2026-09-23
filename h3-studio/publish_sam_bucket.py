"""Publish authorized official SAM weights and license to MissingLink's bucket.

Run only with an HF account approved for facebook/sam3 and bucket write access.
Downloads from Meta first; a pending gate is an error, never bypassed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="MissingLinkBuilder/wheels")
    parser.add_argument("--directory", default="/content/sam3-publish")
    args = parser.parse_args()
    api = HfApi()
    revision = api.model_info("facebook/sam3").sha
    directory = Path(args.directory) / revision
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    # Complete the authorized download before publishing anything.
    for name in ("sam3.pt", "config.json", "LICENSE"):
        path = Path(hf_hub_download("facebook/sam3", name, revision=revision,
                                   local_dir=str(directory)))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries[name] = {"sha256": digest, "size": path.stat().st_size}
    prefix = f"models/sam3/{revision}"
    manifest = {"source": "facebook/sam3", "revision": revision,
                "prefix": prefix, "files": entries,
                "license": "SAM License; redistributed under Meta's original terms"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    api.batch_bucket_files(args.bucket, add=[
        (str(directory / name), f"{prefix}/{name}") for name in entries
    ] + [(str(directory / "manifest.json"), f"{prefix}/manifest.json")])
    # Publish the discovery pointer last, after the complete licensed package.
    api.batch_bucket_files(args.bucket, add=[
        (str(directory / "manifest.json"), "models/sam3/manifest.json")])
    print(f"Published SAM weights, configuration, license and SHA256 manifest to {args.bucket}/{prefix}")


if __name__ == "__main__":
    main()
