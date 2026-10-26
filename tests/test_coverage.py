"""Static instruction-coverage analysis (task T3.1)."""
import struct
from pathlib import Path

import pytest

from ps2toolkit.analysis.coverage import (
    analyze_coverage,
    classify_unsupported,
    load_boot_elf,
    scan_elf,
    render_markdown,
)
from ps2toolkit.elf.ps2elf import PS2ELF
from tests import asm as a

CODE = 0x00100000
PT_LOAD = 1
PF_X = 1
PF_W = 2
PF_R = 4


def build_elf(entry: int, words: list[int], vaddr: int = CODE,
              exec_flags: int = PF_R | PF_X) -> bytes:
    """Minimal little-endian ELF32/MIPS with one PT_LOAD segment."""
    code = b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words)
    ehdr = struct.pack(
        "<16sHHIIIIIHHHHHH",
        b"\x7fELF" + bytes([1, 1, 1]) + bytes(9),  # ident: 32-bit LE, v1
        2,                                        # ET_EXEC
        8,                                        # EM_MIPS
        1,                                        # e_version
        entry,
        52,                                       # e_phoff
        0,                                        # e_shoff
        0,                                        # e_flags
        52, 32, 1, 0, 0, 0,
    )
    phdr = struct.pack("<IIIIIIII", PT_LOAD, 0x100, vaddr, vaddr,
                       len(code), len(code), exec_flags, 0x1000)
    return ehdr + phdr + bytes(0x100 - len(ehdr) - len(phdr)) + code


def scan(words, entry: int = CODE):
    return scan_elf(PS2ELF.from_bytes(build_elf(entry, words)))


def test_straight_line_stops_at_return():
    report = scan([
        a.addiu(a.T0, a.ZERO, 1),
        a.jr(a.RA),
        a.NOP,                      # delay slot: reachable
        a.addiu(a.T1, a.ZERO, 9),   # after the return: not reachable
    ])
    assert report["reachable_words"] == 3
    assert report["code_words"] == 4
    assert report["unsupported_words"] == 0
    assert report["by_mnemonic"]["jr"] == 1


def test_branch_visits_both_edges_and_delay_slot():
    report = scan([
        a.beq(a.T0, a.ZERO, 4),   # @0x00 -> 0x14
        a.addiu(a.T1, a.ZERO, 1),  # @0x04 delay
        a.addiu(a.T2, a.ZERO, 2),  # @0x08 fallthrough
        a.jr(a.RA),                # @0x0c
        a.NOP,                     # @0x10
        a.addiu(a.T3, a.ZERO, 3),  # @0x14 branch target
        a.jr(a.RA),                # @0x18
        a.NOP,                     # @0x1c
    ])
    assert report["reachable_words"] == 8
    assert report["unsupported_words"] == 0


def test_jal_discovers_callee_and_continues_caller():
    report = scan([
        a.jal(CODE + 0x20),        # @0x00
        a.NOP,                     # @0x04 delay
        a.jr(a.RA),                # @0x08
        a.NOP,                     # @0x0c
        a.NOP, a.NOP, a.NOP,       # @0x10..0x18
        a.NOP,                     # @0x1c
        a.addiu(a.V0, a.ZERO, 5),  # @0x20 callee
        a.jr(a.RA),                # @0x24
        a.NOP,                     # @0x28 delay
    ])
    # caller: jal, delay, jr, delay = 4; callee: 3. The NOPs in between are
    # after the caller's return, so they are not reachable.
    assert report["reachable_words"] == 7
    assert report["by_mnemonic"]["jal"] == 1


def test_indirect_jump_recorded_and_walk_stops():
    report = scan([
        a.jr(a.T0),                 # @0x00 jump table dispatch
        a.NOP,                      # @0x04 delay
        a.addiu(a.T1, a.ZERO, 1),   # @0x08 unreachable fallthrough
    ])
    assert report["reachable_words"] == 2
    assert report["indirect_control_count"] == 1
    assert report["indirect_control"][0]["kind"] == "indirect-jump"


def test_indirect_call_continues_caller():
    report = scan([
        a.jalr(a.RA, a.T9),         # @0x00 call through register
        a.NOP,                      # @0x04 delay
        a.jr(a.RA),                 # @0x08 caller continues
        a.NOP,                      # @0x0c delay
    ])
    assert report["reachable_words"] == 4
    assert report["indirect_control_count"] == 1
    assert report["indirect_control"][0]["kind"] == "indirect-call"


def test_unsupported_words_are_counted_by_category():
    report = scan([
        0x70809E28,   # MMI (PADDW)
        0x460062C6,   # COP1/FPU (MOV.S)
        0x68000000,   # LDL (64-bit unaligned load, still unsupported)
        a.jr(a.RA),
        a.NOP,
    ])
    assert report["unsupported_words"] == 3
    assert report["unsupported_categories"] == {
        "mmi": 1, "cop1-fpu": 1, "64-bit-memory": 1,
    }
    assert {s["category"] for s in report["unsupported_samples"]} == {
        "mmi", "cop1-fpu", "64-bit-memory",
    }
    # 3 unsupported of 5 reachable words
    assert report["supported_pct"] == 40.0


@pytest.mark.parametrize("word, category", [
    (0x70809E28, "mmi"),
    (0x7C080000, "mmi"),      # LQ (opcode 0x1E)
    (0x460062C6, "cop1-fpu"),
    (0x42000018, "cop0"),     # ERET
    (0xC4600000, "fpu-memory"),
    (0xD8890000, "vu-memory"),
    (0x0000002D, "64-bit-alu"),
    (0x0000003F, "64-bit-shift"),
    (0x0000000A, "conditional-move"),
    (0x04100000, "regimm"),   # BLTZAL
    (0xBC000000, "cache-pref"),
    (0xFC000000, "64-bit-memory"),
])
def test_classify_unsupported(word, category):
    assert classify_unsupported(word) == category


def test_entry_outside_code_segment_raises():
    with pytest.raises(ValueError, match="not inside an executable segment"):
        scan_elf(PS2ELF.from_bytes(build_elf(0x200000, [a.NOP])))


def test_non_executable_segment_is_ignored():
    elf = PS2ELF.from_bytes(build_elf(CODE, [a.NOP], exec_flags=PF_R | PF_W))
    with pytest.raises(ValueError, match="no executable PT_LOAD"):
        scan_elf(elf)


def test_load_boot_elf_from_elf_file(tmp_path):
    path = tmp_path / "boot.elf"
    path.write_bytes(build_elf(CODE, [a.addiu(a.V0, a.ZERO, 1), a.jr(a.RA), a.NOP]))
    elf, meta = load_boot_elf(path)
    assert elf.entry == CODE
    assert meta["name"] == "boot.elf"
    report = analyze_coverage(path)
    assert report["elf"]["name"] == "boot.elf"
    assert report["reachable_words"] == 3


def test_render_markdown(tmp_path):
    path = tmp_path / "boot.elf"
    path.write_bytes(build_elf(CODE, [a.addiu(a.V0, a.ZERO, 1), a.jr(a.RA), a.NOP]))
    md = render_markdown(analyze_coverage(path))
    assert "Boot ELF Instruction Coverage" in md
    assert f"0x{CODE:08X}" in md
    assert "Unsupported categories" in md


REAL_DISC = Path(__file__).resolve().parents[1] / "Rumble Racing.cue"


@pytest.mark.skipif(not REAL_DISC.exists(), reason="real Rumble Racing disc not present")
def test_real_disc_coverage():
    report = analyze_coverage(REAL_DISC)
    assert report["elf"]["name"].startswith("SLUS_201.74")
    assert report["code_words"] > 200_000
    assert report["reachable_words"] > 100_000
    assert report["unsupported_words"] > 0
