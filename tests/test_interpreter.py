"""Tests for the golden-model EE interpreter (T2.3).

The interpreter is the correctness oracle for the recompiler: it implements
the documented semantics of the supported R5900 subset.
"""
import pytest

from ps2toolkit.golden.interpreter import (
    EEMemory,
    EEState,
    GuestSyscall,
    MemoryFault,
    run,
    step,
)

from tests import asm as a

CODE = 0x00100000
STACK = 0x00110000


def make():
    mem = EEMemory()
    state = EEState(pc=CODE)
    return state, mem


def setu32(x):
    return x & 0xFFFFFFFF


def signed32(x):
    return x - 0x100000000 if x & 0x80000000 else x


# --- ALU semantics ---

def test_addiu_basic():
    st, mem = make()
    st.regs[a.T0] = 5
    a.assemble(mem, CODE, a.addiu(a.T1, a.T0, 0x20), a.NOP)
    step(st, mem)
    assert st.regs[a.T1] == 0x25


def test_addiu_wraps_at_32_bits():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF
    a.assemble(mem, CODE, a.addiu(a.T1, a.T0, 2), a.NOP)
    step(st, mem)
    assert st.regs[a.T1] == 1


def test_addiu_negative_immediate():
    st, mem = make()
    st.regs[a.T0] = 10
    a.assemble(mem, CODE, a.addiu(a.T1, a.T0, -4), a.NOP)
    step(st, mem)
    assert st.regs[a.T1] == 6


def test_reg0_is_always_zero():
    st, mem = make()
    st.regs[a.T0] = 123
    a.assemble(mem, CODE, a.addiu(a.ZERO, a.T0, 0), a.NOP)
    step(st, mem)
    assert st.regs[a.ZERO] == 0


def test_lui_ori_builds_constant():
    st, mem = make()
    a.assemble(mem, CODE, a.lui(a.T0, 0xABCD), a.ori(a.T0, a.T0, 0x1234), a.NOP)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T0] == 0xABCD1234


def test_addu_subu():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF
    st.regs[a.T1] = 2
    a.assemble(mem, CODE, a.addu(a.T2, a.T0, a.T1), a.subu(a.T3, a.T2, a.T1), a.NOP)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == 1
    assert st.regs[a.T3] == 0xFFFFFFFF  # 1 - 2 wraps


def test_logic_ops():
    st, mem = make()
    st.regs[a.T0] = 0xF0F0F0F0
    st.regs[a.T1] = 0x0FF00FF0
    a.assemble(
        mem, CODE,
        a.and_(a.T2, a.T0, a.T1),
        a.or_(a.T3, a.T0, a.T1),
        a.xor_(a.T4, a.T0, a.T1),
        a.nor_(a.T5, a.T0, a.T1),
        a.NOP,
    )
    for _ in range(4):
        step(st, mem)
    assert st.regs[a.T2] == 0x00F000F0
    assert st.regs[a.T3] == 0xFFF0FFF0
    assert st.regs[a.T4] == 0xFF00FF00
    assert st.regs[a.T5] == 0x000F000F


def test_slt_sltu():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF  # -1 signed
    st.regs[a.T1] = 1
    a.assemble(mem, CODE, a.slt(a.T2, a.T0, a.T1), a.sltu(a.T3, a.T0, a.T1), a.NOP)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == 1   # -1 < 1 signed
    assert st.regs[a.T3] == 0   # 0xFFFFFFFF > 1 unsigned


def test_shifts():
    st, mem = make()
    st.regs[a.T0] = 0x80000000
    st.regs[a.T1] = 4
    a.assemble(
        mem, CODE,
        a.sll(a.T2, a.T0, 4),
        a.srl(a.T3, a.T0, 28),
        a.sra(a.T4, a.T0, 28),
        a.sllv(a.T5, a.T0, a.T1),
        a.srlv(a.T6, a.T0, a.T1),
        a.srav(a.T7, a.T0, a.T1),
        a.NOP,
    )
    for _ in range(6):
        step(st, mem)
    assert st.regs[a.T2] == 0x00000000
    assert st.regs[a.T3] == 0x00000008
    assert st.regs[a.T4] == 0xFFFFFFF8  # -8 after arithmetic shift of 0x80000000
    assert st.regs[a.T5] == 0
    assert st.regs[a.T6] == 0x08000000
    assert st.regs[a.T7] == 0xF8000000  # arithmetic shift of 0x80000000 by 4


# --- memory ---

def test_sw_lw_roundtrip_big_endian():
    st, mem = make()
    st.regs[a.T0] = STACK
    st.regs[a.T1] = 0xDEADBEEF
    a.assemble(mem, CODE, a.sw(a.T1, 0, a.T0), a.lw(a.T2, 0, a.T0), a.NOP)
    step(st, mem)
    step(st, mem)
    assert mem.read(STACK, 4) == b"\xDE\xAD\xBE\xEF"  # big-endian storage
    assert st.regs[a.T2] == 0xDEADBEEF


def test_lb_lbu_lh_lhu():
    st, mem = make()
    st.regs[a.T0] = STACK
    mem.write(STACK, b"\x80\xFF\x7F\x01")
    a.assemble(
        mem, CODE,
        a.lb(a.T1, 0, a.T0),
        a.lbu(a.T2, 0, a.T0),
        a.lh(a.T3, 0, a.T0),
        a.lhu(a.T4, 0, a.T0),
        a.NOP,
    )
    for _ in range(4):
        step(st, mem)
    assert st.regs[a.T1] == 0xFFFFFF80  # sign-extended byte
    assert st.regs[a.T2] == 0x80
    assert st.regs[a.T3] == 0xFFFF80FF  # sign-extended halfword
    assert st.regs[a.T4] == 0x80FF


def test_sb_sh():
    st, mem = make()
    st.regs[a.T0] = STACK
    st.regs[a.T1] = 0x1234
    st.regs[a.T2] = 0xAB
    a.assemble(mem, CODE, a.sh(a.T1, 0, a.T0), a.sb(a.T2, 3, a.T0), a.NOP)
    step(st, mem)
    step(st, mem)
    assert mem.read(STACK, 4) == b"\x12\x34\x00\xAB"


def test_unaligned_lwl_lwr_pair():
    # BE unaligned word load at STACK+1: canonical pair lwl rt,X / lwr rt,X+3
    # Memory: 12 34 56 78 9A  -> word at STACK+1 = 34 56 78 9A
    st, mem = make()
    st.regs[a.T0] = STACK
    mem.write(STACK, b"\x12\x34\x56\x78\x9A")
    a.assemble(
        mem, CODE,
        a.lwl(a.T1, 1, a.T0),   # bytes STACK+1..+3 -> high side of T1
        a.lwr(a.T1, 4, a.T0),   # byte  STACK+4      -> low side of T1
        a.NOP,
    )
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T1] == 0x3456789A


def test_lwl_aligned_is_full_word():
    st, mem = make()
    st.regs[a.T0] = STACK
    mem.write(STACK, b"\x12\x34\x56\x78")
    a.assemble(mem, CODE, a.lwl(a.T1, 0, a.T0), a.NOP)
    step(st, mem)
    assert st.regs[a.T1] == 0x12345678


def test_unaligned_swl_swr_pair():
    # mirror of the load pair: swl rt,X / swr rt,X+3 stores BE word at X
    st, mem = make()
    st.regs[a.T0] = STACK
    st.regs[a.T1] = 0x3456789A
    mem.write(STACK, b"\x00" * 6)
    a.assemble(
        mem, CODE,
        a.swl(a.T1, 1, a.T0),
        a.swr(a.T1, 4, a.T0),
        a.NOP,
    )
    step(st, mem)
    step(st, mem)
    assert mem.read(STACK, 5) == b"\x00\x34\x56\x78\x9A"


# --- HI/LO and multip/division ---

def test_mult_signed():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFE  # -2
    st.regs[a.T1] = 3
    a.assemble(mem, CODE, a.mult(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    step(st, mem)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == setu32(-6)
    assert st.regs[a.T3] == 0xFFFFFFFF  # sign-extended upper half


def test_multu():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF
    st.regs[a.T1] = 2
    a.assemble(mem, CODE, a.multu(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(3):
        step(st, mem)
    assert st.regs[a.T2] == 0xFFFFFFFE
    assert st.regs[a.T3] == 0x00000001


def test_div_signed():
    st, mem = make()
    st.regs[a.T0] = setu32(-7)
    st.regs[a.T1] = 2
    a.assemble(mem, CODE, a.div(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(3):
        step(st, mem)
    assert st.regs[a.T2] == setu32(-3)  # truncation toward zero
    assert st.regs[a.T3] == setu32(-1)


def test_div_by_zero_ps2_semantics():
    st, mem = make()
    st.regs[a.T0] = setu32(-5)
    st.regs[a.T1] = 0
    a.assemble(mem, CODE, a.div(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(3):
        step(st, mem)
    assert st.regs[a.T2] == 0x80000000  # LO = 0x80000000 for negative dividend
    assert st.regs[a.T3] == setu32(-5)  # HI = dividend


def test_mthi_mtlo():
    st, mem = make()
    st.regs[a.T0] = 111
    st.regs[a.T1] = 222
    a.assemble(mem, CODE, a.mthi(a.T0), a.mtlo(a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(4):
        step(st, mem)
    assert st.regs[a.T2] == 222
    assert st.regs[a.T3] == 111


# --- control flow ---

def test_beq_taken_executes_delay_slot():
    st, mem = make()
    st.regs[a.T0] = 1
    st.regs[a.T1] = 1
    a.assemble(
        mem, CODE,
        a.beq(a.T0, a.T1, 2),        # -> CODE+12
        a.addiu(a.T2, a.ZERO, 11),   # delay slot: always executes
        a.addiu(a.T3, a.ZERO, 99),   # skipped
        a.addiu(a.T4, a.ZERO, 22),   # target
        a.NOP,
    )
    run(st, mem, max_steps=10, until=CODE + 16)
    assert st.regs[a.T2] == 11
    assert st.regs[a.T3] == 0
    assert st.regs[a.T4] == 22


def test_bne_not_taken_falls_through():
    st, mem = make()
    st.regs[a.T0] = 5
    a.assemble(
        mem, CODE,
        a.beq(a.T0, a.ZERO, 4),     # 5 == 0 is false -> not taken
        a.addiu(a.T1, a.ZERO, 1),   # delay slot executes
        a.addiu(a.T2, a.ZERO, 2),   # executes (not taken)
        a.NOP,
    )
    run(st, mem, max_steps=10, until=CODE + 12)
    assert st.regs[a.T1] == 1
    assert st.regs[a.T2] == 2


def test_blez_bgtz_bltz_bgez():
    st, mem = make()
    st.regs[a.T0] = setu32(-1)
    a.assemble(
        mem, CODE,
        a.blez(a.T0, 2),
        a.NOP,
        a.NOP,
        a.addiu(a.T1, a.ZERO, 1),
        a.NOP,
    )
    run(st, mem, max_steps=10, until=CODE + 16)
    assert st.regs[a.T1] == 1  # -1 <= 0, branch taken


def test_jal_sets_ra_and_jr_returns():
    st, mem = make()
    func = CODE + 0x40
    a.assemble(
        mem, CODE,
        a.jal(func),
        a.NOP,
        a.addiu(a.T9, a.ZERO, 1),  # return site + 8
        a.NOP,
    )
    a.assemble(
        mem, func,
        a.addiu(a.T0, a.ZERO, 7),
        a.jr(a.RA),
        a.NOP,
    )
    run(st, mem, max_steps=20, until=CODE + 0x0C)
    assert st.regs[a.T0] == 7
    assert st.regs[a.RA] == CODE + 8
    assert st.regs[a.T9] == 1


def test_loop_sums_1_to_10():
    # t0 = counter, t1 = sum; loop: addiu t0,1; addu t1,t0; bne t0,10,-2; nop
    st, mem = make()
    a.assemble(
        mem, CODE,
        a.addiu(a.T0, a.ZERO, 0),
        a.addiu(a.T2, a.ZERO, 10),
        a.addiu(a.T0, a.T0, 1),
        a.addu(a.T1, a.T1, a.T0),
        a.bne(a.T0, a.T2, -3),
        a.NOP,
        a.NOP,
    )
    run(st, mem, max_steps=1000, until=CODE + 28)
    assert st.regs[a.T1] == 55
    assert st.regs[a.T0] == 10


def test_syscall_raises_trap():
    st, mem = make()
    a.assemble(mem, CODE, a.syscall(), a.NOP)
    with pytest.raises(GuestSyscall) as exc:
        step(st, mem)
    assert exc.value.code == 0


def test_memory_fault_on_oob():
    st, mem = make()
    st.regs[a.T0] = 0x04000000  # beyond 32MB RAM
    a.assemble(mem, CODE, a.lw(a.T1, 0, a.T0), a.NOP)
    with pytest.raises(MemoryFault):
        step(st, mem)


# --- conditional move / hints ---

def test_movz_movn():
    st, mem = make()
    st.regs[a.T0] = 0x1234          # value
    st.regs[a.T1] = 0               # condition
    st.regs[a.T2] = 0xFFFF          # previous dest
    a.assemble(mem, CODE, a.movz(a.T2, a.T0, a.T1), a.NOP)
    step(st, mem)
    assert st.regs[a.T2] == 0x1234  # rt == 0 -> move

    st, mem = make()
    st.regs[a.T0] = 0x1234
    st.regs[a.T1] = 1
    st.regs[a.T2] = 0xFFFF
    a.assemble(mem, CODE, a.movz(a.T2, a.T0, a.T1), a.NOP)
    step(st, mem)
    assert st.regs[a.T2] == 0xFFFF

    st, mem = make()
    st.regs[a.T0] = 0x1234
    st.regs[a.T1] = 7
    st.regs[a.T2] = 0xFFFF
    a.assemble(mem, CODE, a.movn(a.T2, a.T0, a.T1), a.NOP)
    step(st, mem)
    assert st.regs[a.T2] == 0x1234  # rt != 0 -> move


def test_sync_cache_pref_are_noops():
    st, mem = make()
    a.assemble(
        mem, CODE,
        a.sync(),
        a.cache(3, 16, a.SP),
        a.pref(2, 32, a.SP),
        a.NOP,
    )
    for expected_pc in (CODE + 4, CODE + 8, CODE + 12):
        step(st, mem)
        assert st.pc == expected_pc
    assert st.regs[a.SP] == 0  # hint opcodes do not read or write registers


# --- branch-likely / branch-and-link ---

def test_branch_likely_taken_executes_delay_slot():
    st, mem = make()
    a.assemble(
        mem, CODE,
        a.beql(a.T0, a.T0, 2),       # taken -> CODE+12
        a.addiu(a.T1, a.ZERO, 11),   # delay: executes
        a.addiu(a.T2, a.ZERO, 99),   # skipped
        a.addiu(a.T3, a.ZERO, 22),   # target
    )
    step(st, mem)
    assert st.pc == CODE + 12
    assert st.regs[a.T1] == 11
    assert st.regs[a.T2] == 0


def test_branch_likely_not_taken_nullifies_delay_slot():
    st, mem = make()
    st.regs[a.T0] = 1
    st.regs[a.T1] = 2
    a.assemble(
        mem, CODE,
        a.beql(a.T0, a.T1, 4),       # 1 != 2 -> not taken -> CODE+8
        a.addiu(a.T2, a.ZERO, 11),   # delay: nullified
        a.addiu(a.T3, a.ZERO, 22),   # fallthrough
    )
    steps = step(st, mem)
    assert steps == 1
    assert st.pc == CODE + 8
    assert st.regs[a.T2] == 0
    step(st, mem)
    assert st.regs[a.T3] == 22


def test_branch_and_link_writes_ra_only_when_taken():
    st, mem = make()
    st.regs[a.T0] = 5
    a.assemble(
        mem, CODE,
        a.bgezal(a.T0, 2),           # taken -> CODE+12; ra = CODE+8
        a.addiu(a.T1, a.ZERO, 1),    # delay
        a.addiu(a.T2, a.ZERO, 99),   # skipped
        a.addiu(a.T3, a.ZERO, 2),    # target
    )
    step(st, mem)
    assert st.pc == CODE + 12
    assert st.regs[a.RA] == CODE + 8
    assert st.regs[a.T1] == 1
    assert st.regs[a.T2] == 0

    st, mem = make()
    st.regs[a.T0] = setu32(-1)       # bgezal not taken: link is not written
    st.regs[a.RA] = 0xABCD
    a.assemble(
        mem, CODE,
        a.bgezal(a.T0, 4),
        a.NOP,
        a.NOP,
    )
    step(st, mem)
    assert st.pc == CODE + 8
    assert st.regs[a.RA] == 0xABCD


def test_kseg0_address_translation():
    st, mem = make()
    st.regs[a.T0] = 0xA0000000 | STACK  # uncached kseg1 mirror
    st.regs[a.T1] = 0xCAFEBABE
    a.assemble(mem, CODE, a.sw(a.T1, 0, a.T0), a.NOP)
    step(st, mem)
    assert mem.read(STACK, 4) == b"\xCA\xFE\xBA\xBE"

