"""
One-time setup: create UC catalog, schema, volume, and upload initial files.

Run this before starting the auto-research loop.

Usage:
    python databricks_resources/setup_volumes.py --profile e2-demo-west
"""

import argparse
import io
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="e2-demo-west")
    parser.add_argument("--catalog", type=str, default="main")
    parser.add_argument("--schema", type=str, default="auto_research")
    parser.add_argument("--volume", type=str, default="autoresearch")
    args = parser.parse_args()

    client = WorkspaceClient(profile=args.profile)
    volume_base = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}"

    # Create schema if needed
    try:
        client.schemas.get(f"{args.catalog}.{args.schema}")
        print(f"Schema {args.catalog}.{args.schema} exists")
    except Exception:
        print(f"Creating schema {args.catalog}.{args.schema}")
        client.schemas.create(name=args.schema, catalog_name=args.catalog)

    # Create volume if needed
    full_name = f"{args.catalog}.{args.schema}.{args.volume}"
    try:
        client.volumes.read(full_name)
        print(f"Volume {full_name} exists")
    except Exception:
        print(f"Creating volume {full_name}")
        client.volumes.create(
            catalog_name=args.catalog,
            schema_name=args.schema,
            name=args.volume,
            volume_type=VolumeType.MANAGED,
        )

    # Create subdirectories
    for subdir in ["data", "adapters", "scripts"]:
        path = f"{volume_base}/{subdir}"
        try:
            client.files.create_directory(path)
            print(f"Created directory: {path}")
        except Exception:
            print(f"Directory exists: {path}")

    # Upload train.py
    train_py = Path("train.py")
    if train_py.exists():
        client.files.upload(
            file_path=f"{volume_base}/train.py",
            contents=io.BytesIO(train_py.read_bytes()),
            overwrite=True,
        )
        print(f"Uploaded train.py to {volume_base}/train.py")

    # Upload prepare.py
    prepare_py = Path("prepare.py")
    if prepare_py.exists():
        client.files.upload(
            file_path=f"{volume_base}/prepare.py",
            contents=io.BytesIO(prepare_py.read_bytes()),
            overwrite=True,
        )
        print(f"Uploaded prepare.py to {volume_base}/prepare.py")

    print(f"\nSetup complete. Volume base: {volume_base}")
    print("Next: run prepare.py to download and preprocess financial data")


if __name__ == "__main__":
    main()
