"""Boot tracer: run real/synthetic ELF entries in the golden model (T3.4)."""
from pathlib import Path

import pytest

from ps2toolkit.elf.ps2elf import PS2ELF
from ps2toolkit.golden.boot import load_guest_image, render_markdown, trace_boot
from tests import asm as a
from tests.test_coverage import CODE, build_elf

DATA_ADDR = CODE + 0x40
DATA = bytes(range(0x10, 0x20))


def _program() -> bytes:
    words = [
        a.lui(a.T0, CODE >> 16),
        a.addiu(a.T0, a.T0, 0x40),        # $t0 = data
        a.lq(a.T1, 0, a.T0),              # 16-byte load
        a.sq(a.T1, 0x10, a.T0),           # 16-byte store
        a.addiu(a.V1, a.ZERO, 0x3C),      # PS2 SDK: number in $v1
        a.syscall(),                       # 0x3c is HLE-dispatched here
        a.addiu(a.S0, a.ZERO, 0x55),
        0xD8A90000,                        # LQC2: still unsupported
    ]
    # pad code to 0x40 bytes so DATA_ADDR lands right after it
    words += [a.NOP] * (16 - len(words))
    return build_elf(CODE, words, extra=DATA)


def test_load_guest_image_copies_segments():
    elf = PS2ELF.from_bytes(_program())
    mem = load_guest_image(elf)
    assert mem.read(CODE, 4) == _program()[0x100 : 0x104]
    assert mem.read(DATA_ADDR, len(DATA)) == DATA


def test_trace_lq_sq_syscall_and_unsupported_stop():
    elf = PS2ELF.from_bytes(_program())
    result = trace_boot(elf, max_steps=10_000,
                        syscall_returns={0x3C: 0x1234})
    report = result.report
    assert report["stop_reason"] == "unsupported"
    assert report["detail"]["word"] == 0xD8A90000
    assert report["detail"]["pc"] == CODE + 0x1C
    assert report["syscalls"] == {"0x3c": 1}
    assert result.memory.read(DATA_ADDR + 0x10, 16) == DATA
    # the HLE result landed in $v0, and execution continued past the syscall
    assert report["registers"]["v0"] == 0x1234


def test_trace_uses_v1_as_syscall_number_when_instruction_code_is_zero():
    elf = PS2ELF.from_bytes(_program())
    result = trace_boot(elf, max_steps=10_000, syscall_returns={0x3C: 7})
    assert result.report["syscall_log"][0]["code"] == 0
    assert result.report["syscall_log"][0]["number"] == 0x3C


def test_trace_step_limit():
    elf = PS2ELF.from_bytes(build_elf(CODE, [
        a.beq(a.ZERO, a.ZERO, -1),   # infinite loop
        a.NOP,
    ]))
    result = trace_boot(elf, max_steps=10)
    assert result.report["stop_reason"] == "step-limit"
    assert result.report["steps"] == 10


def test_trace_memory_fault():
    elf = PS2ELF.from_bytes(build_elf(CODE, [
        a.lui(a.T0, 0x0400),         # 64 MiB: beyond guest RAM
        a.lw(a.T1, 0, a.T0),
        a.NOP,
    ]))
    result = trace_boot(elf, max_steps=100)
    assert result.report["stop_reason"] == "memory-fault"
    assert "outside RAM" in result.report["detail"]["message"]


def test_trace_guest_break():
    elf = PS2ELF.from_bytes(build_elf(CODE, [
        a.enc_r(0, 0, 0, 0, 7, 13),  # break code 7
        a.NOP,
    ]))
    result = trace_boot(elf, max_steps=100)
    assert result.report["stop_reason"] == "break"
    assert result.report["detail"]["code"] == 7


def test_render_markdown():
    elf = PS2ELF.from_bytes(_program())
    result = trace_boot(elf, max_steps=1000)
    md = render_markdown(result.report, source="disc.cue", elf_name="SLUS_201.74")
    assert "Boot Trace" in md
    assert "unsupported" in md
    assert "SLUS_201.74" in md


REAL_DISC = Path(__file__).resolve().parents[1] / "Rumble Racing.cue"


@pytest.mark.skipif(not REAL_DISC.exists(), reason="real Rumble Racing disc not present")
def test_real_boot_runs_table_init():
    from ps2toolkit.analysis.coverage import load_boot_elf

    elf, _meta = load_boot_elf(REAL_DISC)
    result = trace_boot(
        elf,
        max_steps=300_000,
        syscall_returns=lambda n, st, mem: st.regs[4] if n == 0x3C else 0,
    )
    report = result.report
    # The CRT clears 0x1eef80..0x24b010 with SQ, runs its init calls, and
    # spends the rest of the budget generating 16K-entry sin/cos tables
    # (0x163FF8..0x16402C). No unsupported instruction is reached.
    assert report["stop_reason"] == "step-limit"
    assert report["steps"] == 300_000
    assert report["unique_pcs"] > 400
    assert any(h["pc"] == 0x00163FF8 for h in report["hot_pcs"])
    assert "0x3c" in report["syscalls"]
    assert "0x3e" in report["syscalls"]
