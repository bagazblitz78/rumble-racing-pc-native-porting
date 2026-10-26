"""Static instruction-coverage analysis of PS2 boot executables (task T3.1).

The recompiler front-end currently supports a documented MVP subset of the
R5900 ISA. Before extending it, we need to know which instructions the real
game actually executes and how often - not just which words appear in a
linear sweep of .text (which also covers jump tables and constants).

This module therefore walks the executable PT_LOAD segments by recursive
descent, starting at the ELF entry point and following:

- both edges of conditional branches (including the delay slot),
- `j` targets,
- `jal` targets (callees), while continuing after the call,
- `jalr` callees are recorded as indirect (target unknown),
- `jr $ra` ends a walk (function return).

Reachable words that the decoder classifies as "unsupported" are grouped by
category (MMI, COP1/FPU, COP0, 64-bit integer, ...) so the next decoder
extension can be prioritised by real frequency.

`load_boot_elf()` reads the boot executable straight out of a cue/bin/iso
(the same path the disc analysis report uses), so the analysis can be run
against an untouched disc image.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..disc.cue import parse_cue, resolve_bin_path
from ..disc.iso9660 import ISO9660
from ..disc.sector import RawDiscImage
from ..disc.systemcnf import find_boot_elf
from ..elf.ps2elf import PS2ELF
from ..r5900.decoder import decode

PF_X = 1
PT_LOAD = 1

_OPCODE_CATEGORY = {
    0x10: "cop0",
    0x11: "cop1-fpu",
    0x12: "cop2-vu0",
    0x1A: "64-bit-memory",   # LDL
    0x1B: "64-bit-memory",   # LDR
    0x1C: "mmi",             # R5900 multimedia instructions
    0x1E: "mmi",             # LQ
    0x1F: "mmi",             # SQ
    0x27: "64-bit-memory",   # LWU
    0x2C: "64-bit-memory",   # SDL
    0x2D: "64-bit-memory",   # SDR
    0x2F: "cache-pref",
    0x31: "fpu-memory",      # LWC1
    0x33: "cache-pref",      # PREF
    0x35: "fpu-memory",      # LDC1
    0x36: "vu-memory",       # LQC2
    0x37: "64-bit-memory",   # LD
    0x39: "fpu-memory",      # SWC1
    0x3D: "fpu-memory",      # SDC1
    0x3E: "vu-memory",       # SQC2
    0x3F: "64-bit-memory",   # SD
}

_SPECIAL_CATEGORY = {
    0x0A: "conditional-move",  # movz
    0x0B: "conditional-move",  # movn
    0x0F: "sync",              # sync
    0x14: "64-bit-shift",      # dsllv
    0x16: "64-bit-shift",      # dsrlv
    0x17: "64-bit-shift",      # dsrav
    0x2C: "64-bit-alu",        # dadd
    0x2D: "64-bit-alu",        # daddu
    0x2E: "64-bit-alu",        # dsub
    0x2F: "64-bit-alu",        # dsubu
    0x30: "trap",  # tge
    0x31: "trap",  # tgeu
    0x32: "trap",  # tlt
    0x33: "trap",  # tltu
    0x34: "trap",  # teq
    0x36: "trap",  # tne
    0x38: "64-bit-shift",      # dsll
    0x3A: "64-bit-shift",      # dsrl
    0x3B: "64-bit-shift",      # dsra
    0x3C: "64-bit-shift",      # dsll32
    0x3E: "64-bit-shift",      # dsrl32
    0x3F: "64-bit-shift",      # dsra32
}


def classify_unsupported(word: int) -> str:
    """Group a word the decoder rejected into an ISA-extension category."""
    opcode = (word >> 26) & 0x3F
    if opcode in _OPCODE_CATEGORY:
        return _OPCODE_CATEGORY[opcode]
    if opcode == 0x00:
        return _SPECIAL_CATEGORY.get(word & 0x3F, "special-other")
    if opcode == 0x01:
        return "regimm"
    return "other"


class CodeMap:
    """Address -> word lookup over the executable PT_LOAD segments."""

    def __init__(self, elf: PS2ELF):
        self.regions: list[tuple[int, int, int, int]] = []  # vaddr, end, offset, index
        self._segments: list[tuple[int, bytes]] = []
        for seg in elf.segments:
            if seg.type == PT_LOAD and (seg.flags & PF_X):
                blob = elf.data[seg.offset : seg.offset + seg.filesz]
                if len(blob) != seg.filesz:
                    continue  # truncated segment: nothing trustworthy to scan
                self._segments.append((seg.vaddr, blob))
                self.regions.append(
                    (seg.vaddr, seg.vaddr + len(blob), seg.offset, len(self._segments) - 1)
                )

    def __bool__(self) -> bool:
        return bool(self._segments)

    def word(self, addr: int) -> int | None:
        for start, end, _off, idx in self.regions:
            if start <= addr and addr + 4 <= end:
                base, blob = self._segments[idx]
                off = addr - base
                return int.from_bytes(blob[off : off + 4], "little")
        return None

    def segment_index(self, addr: int) -> int | None:
        for start, end, _off, idx in self.regions:
            if start <= addr < end:
                return idx
        return None

    def sizes(self) -> list[int]:
        return [len(blob) // 4 for _base, blob in self._segments]

    def bases(self) -> list[int]:
        return [base for base, _blob in self._segments]


def _apply_delay_slot(addr: int, work: list[tuple[int, bool]]) -> None:
    """The delay slot executes on every path; scan only that one word.

    It must not start a linear walk: the instruction after the delay slot
    belongs to whichever edge the branch/jump actually takes.
    """
    work.append((addr + 4, True))


def scan_elf(elf: PS2ELF, sample_limit: int = 200) -> dict:
    """Recursive-descent decode of the ELF's executable segments."""
    code = CodeMap(elf)
    if not code:
        raise ValueError("ELF has no executable PT_LOAD segment")

    entry = elf.entry
    if code.word(entry) is None:
        raise ValueError(f"entry point {entry:#010x} is not inside an executable segment")

    visited: set[int] = set()
    counts: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    samples: list[dict] = []
    indirect: list[dict] = []
    work: list[tuple[int, bool]] = [(entry, False)]

    while work:
        addr, single = work.pop()
        while True:
            if addr in visited:
                break
            word = code.word(addr)
            if word is None:
                break
            visited.add(addr)
            insn = decode(word, addr)

            if insn.mnemonic == "unsupported":
                category = classify_unsupported(word)
                unsupported[category] += 1
                if len(samples) < sample_limit:
                    samples.append({
                        "addr": addr,
                        "word": word,
                        "category": category,
                        "opcode": insn.opcode,
                    })
            counts[insn.mnemonic] += 1

            if single:
                break  # delay slot: never flows into the next word

            if insn.is_branch:
                target = addr + 4 + insn.imm * 4
                _apply_delay_slot(addr, work)
                work.append((target, False))
                addr += 8
            elif insn.mnemonic in ("j", "jal"):
                _apply_delay_slot(addr, work)
                work.append((insn.target, False))
                if insn.mnemonic == "jal":
                    addr += 8  # the callee returns; keep walking the caller
                else:
                    break
            elif insn.is_indirect_jump:
                _apply_delay_slot(addr, work)
                if insn.rs == 31:
                    break  # jr $ra / jalr $ra: function return
                if insn.mnemonic == "jalr":
                    # call through a register: target unknown, but the caller
                    # continues after the delay slot
                    indirect.append({
                        "addr": addr,
                        "word": word,
                        "kind": "indirect-call",
                        "reg": insn.rs,
                    })
                    addr += 8
                else:
                    indirect.append({
                        "addr": addr,
                        "word": word,
                        "kind": "indirect-jump",
                        "reg": insn.rs,
                    })
                    break
            else:
                addr += 4

    total_words = sum(code.sizes())
    reachable = len(visited)
    unsupported_total = sum(unsupported.values())
    supported = reachable - unsupported_total
    bases = code.bases()
    sizes = code.sizes()

    per_segment = []
    for idx, (base, size) in enumerate(zip(bases, sizes)):
        end = base + size * 4
        seg_reachable = sum(1 for a in visited if base <= a < end)
        per_segment.append({
            "vaddr": base,
            "words": size,
            "reachable": seg_reachable,
            "unreachable": size - seg_reachable,
        })

    return {
        "entry": entry,
        "code_words": total_words,
        "reachable_words": reachable,
        "unreachable_words": total_words - reachable,
        "supported_words": supported,
        "unsupported_words": unsupported_total,
        "supported_pct": round(100.0 * supported / reachable, 2) if reachable else 0.0,
        "segments": per_segment,
        "by_mnemonic": dict(counts.most_common()),
        "unsupported_categories": dict(unsupported.most_common()),
        "unsupported_samples": samples,
        "indirect_control": indirect,
        "indirect_control_count": len(indirect),
    }


def load_boot_elf(path: str | Path) -> tuple[PS2ELF, dict]:
    """Read the boot executable from a disc image, or an ELF file directly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in (".elf", ".prx"):
        data = path.read_bytes()
        return PS2ELF.from_bytes(data), {"source": str(path), "name": path.name,
                                         "size": len(data), "offset": 0}

    # Imported lazily to keep the disc helpers optional for ELF-only use.
    from .report import detect_sector_size

    if path.suffix.lower() == ".cue":
        cue = parse_cue(path)[0]
        bin_path = resolve_bin_path(path, cue.filename)
        track = next((t for t in cue.tracks if t.mode != "AUDIO"), None)
        sector_size = track.sector_size if track else detect_sector_size(bin_path)
    else:
        bin_path = path
        sector_size = detect_sector_size(bin_path)

    with RawDiscImage(bin_path, sector_size) as disc:
        iso = ISO9660(disc)
        entry, offset = find_boot_elf(iso)
        data = iso.read_file(entry)
        meta = {
            "source": str(path),
            "image": str(bin_path),
            "name": entry.name,
            "size": entry.size,
            "offset": offset,
        }
    return PS2ELF.from_bytes(data), meta


def analyze_coverage(path: str | Path, sample_limit: int = 200) -> dict:
    """Load a boot ELF (disc image or .elf) and return a JSON-ready scan."""
    elf, meta = load_boot_elf(path)
    report = scan_elf(elf, sample_limit=sample_limit)
    report["elf"] = {
        **meta,
        "entry": elf.entry,
        "segments": len(elf.segments),
    }
    return report


def render_markdown(report: dict, top: int = 20) -> str:
    """Render a coverage report as Markdown."""
    elf = report["elf"]
    lines = [
        "# Boot ELF Instruction Coverage",
        "",
        f"- **Source**: `{elf['source']}`",
        f"- **Boot ELF**: `{elf['name']}`",
        f"- **Entry**: `0x{report['entry']:08X}`",
        f"- **Executable words**: {report['code_words']:,}",
        f"- **Reachable words (recursive descent)**: {report['reachable_words']:,}",
        f"- **Decoded by MVP front-end**: {report['supported_words']:,}"
        f" ({report['supported_pct']}%)",
        f"- **Unsupported words**: {report['unsupported_words']:,}",
        f"- **Indirect control transfers**: {report['indirect_control_count']:,}",
        "",
        "## Executable segments",
        "",
        "| Base | Words | Reachable | Unreachable |",
        "|---|---|---|---|",
    ]
    for seg in report["segments"]:
        lines.append(
            f"| `0x{seg['vaddr']:08X}` | {seg['words']:,} | "
            f"{seg['reachable']:,} | {seg['unreachable']:,} |"
        )

    lines += ["", "## Unsupported categories", "",
              "| Category | Words |", "|---|---|"]
    for name, n in report["unsupported_categories"].items():
        lines.append(f"| {name} | {n:,} |")

    lines += ["", f"## Top mnemonics (of {len(report['by_mnemonic'])})", "",
              "| Mnemonic | Words |", "|---|---|"]
    for name, n in list(report["by_mnemonic"].items())[:top]:
        lines.append(f"| `{name}` | {n:,} |")

    lines += ["", "## Unsupported samples", "",
              "| Address | Word | Category |", "|---|---|---|"]
    for s in report["unsupported_samples"][:top]:
        lines.append(
            f"| `0x{s['addr']:08X}` | `0x{s['word']:08X}` | {s['category']} |"
        )

    lines += ["", "## Indirect control transfers", ""]
    if report["indirect_control"]:
        lines += ["| Address | Word | Kind | Register |", "|---|---|---|---|"]
        for s in report["indirect_control"][:top]:
            lines.append(
                f"| `0x{s['addr']:08X}` | `0x{s['word']:08X}` | {s['kind']} | "
                f"`${s['reg']}` |"
            )
        if len(report["indirect_control"]) > top:
            lines.append(
                f"\n_({len(report['indirect_control']) - top} more not shown)_"
            )
    else:
        lines.append("None recorded.")

    lines += [
        "",
        "## Implications for the recompiler",
        "",
        "- **VU0/COP2** is the only large undecoded block left (1,937 VU "
        "memory forms + 1,454 macro-mode ops), plus 44 unaligned 64-bit "
        "forms. The scalar front-end now decodes almost the whole boot path.",
        "- **FPU memory forms** (LWC1/SWC1/LDC1/SDC1) travel with COP1 and need "
        "the same FPU state hand-off.",
        "- **64-bit integer forms** are translated as of T3.3: GPRs hold "
        "64-bit values, 32-bit results sign-extend, and DADDU/DSUB/DADDIU, "
        "the DSLL/DSRL family and LD/SD/LWU all run natively. Only the "
        "unaligned LDL/LDR/SDL/SDR forms remain.",
        "- **COP2/VU0** macro-mode and LQC2/SQC2 need VU0 register state and "
        "will likely be stubbed behind an HLE boundary first.",
        "- **Native codegen is now the gap, not decoding.** MMI, LQ/SQ and "
        "COP1 run in the golden model but still need SSE2 lowering in "
        "x64emit before the recompiler can execute the same code.",
        "- **Indirect control transfers** shown above are jump tables and "
        "function pointers; static recompilation must resolve them (or fall "
        "back to a dispatch stub) before the whole image is reachable.",
    ]
    return "\n".join(lines) + "\n"
