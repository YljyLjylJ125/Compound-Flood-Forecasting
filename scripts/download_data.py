#!/usr/bin/env python3
"""Download and verify the public SF2Bench archive from Harvard Dataverse."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
from urllib.request import urlopen


FILE_ID = 11275874
URL = f"https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}"
EXPECTED_BYTES = 2_136_442_311
EXPECTED_MD5 = "0e82b123b27d2aa64d941812ed2cf709"


def checksum(path: Path) -> str:
    digest = hashlib.md5()  # Dataverse publishes MD5 for this archive.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/sf2bench")
    parser.add_argument("--archive", default="dataset.tar-1.gz")
    parser.add_argument("--remove-archive", action="store_true")
    args = parser.parse_args()
    archive = Path(args.archive)
    output = Path(args.output_root)

    if not archive.exists():
        print(f"Downloading {URL} -> {archive}", flush=True)
        with urlopen(URL) as response, archive.open("wb") as handle:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
    if archive.stat().st_size != EXPECTED_BYTES or checksum(archive) != EXPECTED_MD5:
        raise RuntimeError("SF2Bench archive size/checksum mismatch")

    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        output_resolved = output.resolve()
        for member in bundle.getmembers():
            destination = (output / member.name).resolve()
            if output_resolved not in destination.parents and destination != output_resolved:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(output)
    if args.remove_archive:
        archive.unlink()
    print(f"SF2Bench extracted under {output.resolve()}")


if __name__ == "__main__":
    main()
