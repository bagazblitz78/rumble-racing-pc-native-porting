"""SYSTEM.CNF parsing and boot-ELF location (task T1.5)."""
from __future__ import annotations

from .iso9660 import ISO9660, ISOFile

ELF_MAGIC = b"\x7fELF"
# Some PS2 discs prefix the ELF with a 0x800-byte header inside the file
SUSPECTED_ELF_OFFSETS = (0, 0x800)


def parse_system_cnf(text: str) -> dict[str, str]:
    """Parse KEY = VALUE lines from a SYSTEM.CNF file."""
    config: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        config[key.strip().upper()] = value.strip()
    return config


def strip_disc_path(boot_value: str) -> str:
    """'cdrom0:\\SLUS_201.74;1' -> 'SLUS_201.74;1' (forward slashes kept)."""
    value = boot_value.strip()
    if ":" in value:
        value = value.split(":", 1)[1]
    return value.replace("\\", "/").lstrip("/")


def find_boot_elf(iso: ISO9660) -> tuple[ISOFile, int]:
    """Locate the boot ELF via SYSTEM.CNF.

    Returns (ISOFile of the boot executable, byte offset of the ELF magic
    within that file).
    """
    cnf_entry = None
    for entry in iso.list_dir(iso.root):
        if entry.name.upper().startswith("SYSTEM.CNF"):
            cnf_entry = entry
            break
    if cnf_entry is None:
        raise FileNotFoundError("SYSTEM.CNF not found in disc root")
    cfg = parse_system_cnf(iso.read_file(cnf_entry).decode("latin-1", errors="replace"))
    boot = cfg.get("BOOT2") or cfg.get("BOOT")
    if not boot:
        raise ValueError("SYSTEM.CNF has no BOOT2/BOOT entry")
    elf_entry = iso.resolve(strip_disc_path(boot))
    data = iso.read_file(elf_entry)
    for offset in SUSPECTED_ELF_OFFSETS:
        if data[offset : offset + 4] == ELF_MAGIC:
            return elf_entry, offset
    raise ValueError(f"{elf_entry.name}: ELF magic not found at {SUSPECTED_ELF_OFFSETS}")
