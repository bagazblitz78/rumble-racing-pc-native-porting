"""Disc analysis report generation (task T1.6)."""
from __future__ import annotations

from pathlib import Path

from ..disc.cue import parse_cue, resolve_bin_path
from ..disc.iso9660 import ISO9660
from ..disc.sector import RAW_SECTOR_SIZE, ISO_SECTOR_SIZE, RawDiscImage
from ..disc.systemcnf import find_boot_elf


def detect_sector_size(path: Path) -> int:
    size = path.stat().st_size
    if size % RAW_SECTOR_SIZE == 0:
        return RAW_SECTOR_SIZE
    if size % ISO_SECTOR_SIZE == 0:
        return ISO_SECTOR_SIZE
    raise ValueError(f"{path}: size {size} matches neither 2352 nor 2048 sector layout")


def analyze_image(path: str | Path) -> dict:
    """Analyze a disc image (.cue or raw/.iso) and return a JSON-ready report."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".cue":
        cue = parse_cue(path)[0]
        bin_path = resolve_bin_path(path, cue.filename)
        track = next((t for t in cue.tracks if t.mode != "AUDIO"), None)
        sector_size = track.sector_size if track else detect_sector_size(bin_path)
        cue_info = {
            "cue": path.name,
            "referenced_bin": cue.filename,
            "resolved_bin": bin_path.name,
            "tracks": [
                {"number": t.number, "mode": t.mode, "sector_size": t.sector_size}
                for t in cue.tracks
            ],
        }
    else:
        bin_path = path
        sector_size = detect_sector_size(bin_path)
        cue_info = None

    with RawDiscImage(bin_path, sector_size) as disc:
        iso = ISO9660(disc)
        files = [
            {"name": name, "lba": e.lba, "size": e.size, "is_dir": e.is_dir}
            for name, e in iso.walk()
        ]
        report: dict = {
            "source": str(path),
            "image": str(bin_path),
            "cue": cue_info,
            "sector_size": sector_size,
            "sector_count": disc.sector_count,
            "volume_id": iso.volume_id,
            "system_id": iso.system_id,
            "playstation_disc": iso.is_playstation,
            "block_size": iso.block_size,
            "files": files,
            "boot_elf": None,
            "boot_elf_error": None,
        }
        try:
            entry, offset = find_boot_elf(iso)
            report["boot_elf"] = {
                "name": entry.name,
                "lba": entry.lba,
                "size": entry.size,
                "offset": offset,
            }
        except (FileNotFoundError, ValueError) as exc:
            report["boot_elf_error"] = str(exc)
    return report


def render_markdown(report: dict) -> str:
    """Render an analysis report as Markdown."""
    lines = [
        "# Disc Analysis Report",
        "",
        f"- **Source**: `{report['source']}`",
        f"- **Image**: `{report['image']}`",
        f"- **Sectors**: {report['sector_count']} × {report['sector_size']} bytes",
        f"- **Volume ID**: {report['volume_id']!r}",
        f"- **System ID**: {report['system_id']!r}",
        f"- **PS2 disc (PLAYSTATION marker)**: {report['playstation_disc']}",
    ]
    if report["cue"]:
        c = report["cue"]
        lines.append(
            f"- **Cue**: `{c['cue']}` (references `{c['referenced_bin']}`, resolved to `{c['resolved_bin']}`)"
        )
    lines += ["", "## Boot executable", ""]
    if report["boot_elf"]:
        b = report["boot_elf"]
        lines += [
            f"- **Name**: `{b['name']}`",
            f"- **LBA**: {b['lba']}",
            f"- **Size**: {b['size']} bytes",
            f"- **ELF offset in file**: 0x{b['offset']:X}",
        ]
    else:
        lines.append(f"- **Error**: {report['boot_elf_error']}")
    lines += ["", "## Files", "", "| Name | LBA | Size | Dir |", "|---|---|---|---|"]
    for f in report["files"]:
        lines.append(f"| `{f['name']}` | {f['lba']} | {f['size']} | {'yes' if f['is_dir'] else ''} |")
    return "\n".join(lines) + "\n"
