"""Cue sheet parsing and bin-file resolution (task T1.1).

Handles the common PS2 rip layout: one FILE entry referencing a raw
MODE2/2352 binary, with a filename that may not exactly match the file
on disk (e.g. cue says "Game (USA).bin" but the rip ships "Game.bin").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Sector size implied by the track mode suffix (e.g. "MODE2/2352" -> 2352)
_SECTOR_SIZE = {
    "MODE1/2048": 2048,
    "MODE1/2352": 2352,
    "MODE2/2048": 2048,
    "MODE2/2336": 2336,
    "MODE2/2352": 2352,
}

_FILE_RE = re.compile(r'^FILE\s+"([^"]+)"\s+(\S+)', re.IGNORECASE)
_TRACK_RE = re.compile(r"^\s*TRACK\s+(\d+)\s+(\S+)", re.IGNORECASE)
_INDEX_RE = re.compile(r"^\s*INDEX\s+(\d+)\s+(\d+):(\d+):(\d+)", re.IGNORECASE)


@dataclass
class CueTrack:
    number: int
    mode: str
    sector_size: int
    indexes: dict[int, str] = field(default_factory=dict)


@dataclass
class CueFile:
    filename: str
    file_type: str
    tracks: list[CueTrack] = field(default_factory=list)


def msf_to_frames(msf: str) -> int:
    """Convert an MM:SS:FF time string to an absolute frame count (75 frames/s)."""
    mm, ss, ff = (int(part) for part in msf.split(":"))
    return (mm * 60 + ss) * 75 + ff


def parse_cue(path: str | Path) -> list[CueFile]:
    """Parse a .cue sheet into a list of CueFile entries."""
    path = Path(path)
    files: list[CueFile] = []
    current: CueFile | None = None
    track: CueTrack | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.upper().startswith("REM"):
            continue
        m = _FILE_RE.match(line)
        if m:
            current = CueFile(filename=m.group(1), file_type=m.group(2).upper())
            files.append(current)
            track = None
            continue
        m = _TRACK_RE.match(line)
        if m:
            if current is None:
                raise ValueError(f"{path}: TRACK before FILE")
            mode = m.group(2).upper()
            sector_size = _SECTOR_SIZE.get(mode)
            if sector_size is None and mode != "AUDIO":
                raise ValueError(f"{path}: unknown track mode {mode!r}")
            track = CueTrack(
                number=int(m.group(1)),
                mode=mode,
                sector_size=sector_size or 2352,  # AUDIO defaults to raw 2352
            )
            current.tracks.append(track)
            continue
        m = _INDEX_RE.match(line)
        if m:
            if track is None:
                raise ValueError(f"{path}: INDEX before TRACK")
            idx = int(m.group(1))
            track.indexes[idx] = f"{m.group(2)}:{m.group(3)}:{m.group(4)}"
            continue
    if not files:
        raise ValueError(f"{path}: no FILE entry found")
    return files


def resolve_bin_path(cue_path: str | Path, filename: str) -> Path:
    """Resolve the binary referenced by a cue to a real file on disk.

    Resolution order:
      1. Exact filename next to the cue.
      2. Any single .bin file next to the cue (rips frequently rename the bin).
      3. Raise FileNotFoundError.
    """
    cue_path = Path(cue_path)
    directory = cue_path.parent
    exact = directory / filename
    if exact.is_file():
        return exact
    bins = sorted(directory.glob("*.bin"))
    if len(bins) == 1:
        return bins[0]
    raise FileNotFoundError(
        f"Cannot resolve {filename!r} referenced by {cue_path} "
        f"(found {len(bins)} .bin files in {directory})"
    )
