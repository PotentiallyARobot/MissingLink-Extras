"""Publish licensed public SAM mirrors to MissingLink's bucket.

Public reads need no token. Publishing uses the normal HF token with bucket
write access. Original license and exact mirror revision accompany every file.
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
    parser.add_argument("--model", choices=("sam3", "sam3.1"), default="sam3")
    args = parser.parse_args()
    api = HfApi()
    source = f"AEmotionStudio/{args.model}"
    info = HfApi(token=False).model_info(source, files_metadata=True)
    revision = info.sha
    directory = Path(args.directory) / args.model / revision
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    checkpoint = "sam3.safetensors" if args.model == "sam3" else "sam3.1_multiplex.safetensors"
    # Complete public downloads and validate Hub hashes before any publication.
    for name in (checkpoint, "config.json", "LICENSE"):
        path = Path(hf_hub_download(source, name, revision=revision,
                                   local_dir=str(directory), token=False))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = next(item for item in info.siblings if item.rfilename == name)
        if metadata.lfs and digest != metadata.lfs.sha256:
            raise ValueError(f"Source checksum mismatch: {name}")
        entries[name] = {"sha256": digest, "size": path.stat().st_size}
    if args.model == "sam3":
        # Official image builder expects torch format. Read only safe tensors,
        # retain the original keys/values, and package the compatible checkpoint.
        import torch
        from safetensors.torch import load_file
        weights = load_file(str(directory / checkpoint), device="cpu")
        if not any(key.startswith("detector.") for key in weights):
            raise ValueError("Mirror is not an original-format SAM image checkpoint")
        converted = directory / "sam3.pt"
        torch.save(weights, converted)
        del weights
        with converted.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries["sam3.pt"] = {"sha256": digest, "size": converted.stat().st_size}
    prefix = f"models/{args.model}/{revision}"
    manifest = {"source": f"facebook/{args.model}", "download_source": source,
                "revision": revision, "checkpoint": "sam3.pt" if args.model == "sam3" else checkpoint,
                "prefix": prefix, "files": entries,
                "license": "SAM License; redistributed under Meta's original terms"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    api.batch_bucket_files(args.bucket, add=[
        (str(directory / name), f"{prefix}/{name}") for name in entries
    ] + [(str(directory / "manifest.json"), f"{prefix}/manifest.json")])
    # Publish the discovery pointer last, after the complete licensed package.
    api.batch_bucket_files(args.bucket, add=[
        (str(directory / "manifest.json"), f"models/{args.model}/manifest.json")])
    print(f"Published SAM weights, configuration, license and SHA256 manifest to {args.bucket}/{prefix}")


if __name__ == "__main__":
    main()
