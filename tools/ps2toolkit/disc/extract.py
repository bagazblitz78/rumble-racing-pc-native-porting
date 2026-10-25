"""Safe extraction of files from parsed disc images (task T1.4)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .iso9660 import ISO9660, ISOFile

_UNSAFE_CHARS = re.compile(r'[<>:"|?*\\]')


def _safe_dest_name(name: str, dest_dir: Path) -> Path:
    """Map an ISO entry name to a safe path under dest_dir, rejecting traversal."""
    # ISO names must not contain path separators; strip version suffix stays as-is
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"unsafe ISO entry name: {name!r}")
    candidate = (dest_dir / name).resolve()
    dest_resolved = dest_dir.resolve()
    if not str(candidate).startswith(str(dest_resolved) + str(Path("\\"))) and candidate != dest_resolved:
        raise ValueError(f"entry escapes destination: {name!r}")
    return dest_dir / name


def safe_extract(iso: ISO9660, iso_path: str, dest_dir: str | Path) -> Path:
    """Extract one file (by disc path) to dest_dir. Returns the written path."""
    dest_dir = Path(dest_dir)
    entry = iso.resolve(iso_path)
    if entry.is_dir:
        raise IsADirectoryError(f"{iso_path} is a directory")
    dest = _safe_dest_name(entry.name, dest_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(iso.read_file(entry))
    return dest


def safe_extract_tree(iso: ISO9660, dest_dir: str | Path) -> list[Path]:
    """Extract every file in the image under dest_dir, preserving directories.

    Returns the list of files written.
    """
    dest_dir = Path(dest_dir)
    written: list[Path] = []
    for path, entry in iso.walk():
        # replicate directory structure under dest_dir
        parts = [p for p in path.split("/") if p]
        local = dest_dir.joinpath(*parts)
        if entry.is_dir:
            local.mkdir(parents=True, exist_ok=True)
            continue
        dest = _safe_dest_name(parts[-1], local.parent)
        local.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(iso.read_file(entry))
        written.append(dest)
    return written


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file (used to prove source images stay unchanged)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
