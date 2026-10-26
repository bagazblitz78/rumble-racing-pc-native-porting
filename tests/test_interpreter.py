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
    sext32,
    sext64,
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
    # LUI sign-extends, and ORI is a 64-bit logical op on MIPS64/R5900
    assert st.regs[a.T0] == sext32(0xABCD1234)


def test_addu_subu():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF
    st.regs[a.T1] = 2
    a.assemble(mem, CODE, a.addu(a.T2, a.T0, a.T1), a.subu(a.T3, a.T2, a.T1), a.NOP)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == 1
    assert st.regs[a.T3] == sext32(-1)  # 1 - 2 wraps, sign-extended


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
    assert st.regs[a.T5] == sext64(0xFFFFFFFF000F000F)  # 64-bit NOR


def test_slt_sltu():
    st, mem = make()
    st.regs[a.T0] = sext32(-1)  # -1 signed
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
    assert st.regs[a.T4] == sext32(-8)  # arithmetic shift of 0x80000000
    assert st.regs[a.T5] == 0
    assert st.regs[a.T6] == 0x08000000
    assert st.regs[a.T7] == sext32(0xF8000000)  # arithmetic shift by 4


# --- memory ---

def test_sw_lw_roundtrip_little_endian():
    st, mem = make()
    st.regs[a.T0] = STACK
    st.regs[a.T1] = 0xDEADBEEF
    a.assemble(mem, CODE, a.sw(a.T1, 0, a.T0), a.lw(a.T2, 0, a.T0), a.NOP)
    step(st, mem)
    step(st, mem)
    assert mem.read(STACK, 4) == b"\xEF\xBE\xAD\xDE"  # little-endian storage
    assert st.regs[a.T2] == sext32(0xDEADBEEF)  # LW sign-extends


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
    assert st.regs[a.T1] == sext32(0xFFFFFF80)  # sign-extended byte
    assert st.regs[a.T2] == 0x80
    assert st.regs[a.T3] == sext32(0xFFFFFF80)  # LE halfword 0xFF80, sign-extended
    assert st.regs[a.T4] == 0xFF80


def test_sb_sh():
    st, mem = make()
    st.regs[a.T0] = STACK
    st.regs[a.T1] = 0x1234
    st.regs[a.T2] = 0xAB
    a.assemble(mem, CODE, a.sh(a.T1, 0, a.T0), a.sb(a.T2, 3, a.T0), a.NOP)
    step(st, mem)
    step(st, mem)
    assert mem.read(STACK, 4) == b"\x34\x12\x00\xAB"


def test_unaligned_lwl_lwr_pair():
    # LE unaligned word load at STACK+1: canonical pair lwl rt,X / lwr rt,X+3
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
    assert st.regs[a.T1] == 0x7856349A


def test_lwl_aligned_is_full_word():
    st, mem = make()
    st.regs[a.T0] = STACK
    mem.write(STACK, b"\x12\x34\x56\x78")
    a.assemble(mem, CODE, a.lwl(a.T1, 0, a.T0), a.NOP)
    step(st, mem)
    assert st.regs[a.T1] == 0x78563412


def test_unaligned_swl_swr_pair():
    # mirror of the load pair: swl rt,X / swr rt,X+3 stores LE word at X
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
    assert mem.read(STACK, 5) == b"\x00\x78\x56\x34\x9A"


# --- HI/LO and multip/division ---

def test_mult_signed():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFE  # -2
    st.regs[a.T1] = 3
    a.assemble(mem, CODE, a.mult(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    step(st, mem)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == sext32(-6)
    assert st.regs[a.T3] == sext32(-1)  # sign-extended upper half


def test_multu():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF
    st.regs[a.T1] = 2
    a.assemble(mem, CODE, a.multu(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(3):
        step(st, mem)
    assert st.regs[a.T2] == sext32(0xFFFFFFFE)
    assert st.regs[a.T3] == 0x00000001


def test_div_signed():
    st, mem = make()
    st.regs[a.T0] = setu32(-7)
    st.regs[a.T1] = 2
    a.assemble(mem, CODE, a.div(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(3):
        step(st, mem)
    assert st.regs[a.T2] == sext32(-3)  # truncation toward zero
    assert st.regs[a.T3] == sext32(-1)


def test_div_by_zero_ps2_semantics():
    st, mem = make()
    st.regs[a.T0] = setu32(-5)
    st.regs[a.T1] = 0
    a.assemble(mem, CODE, a.div(a.T0, a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(3):
        step(st, mem)
    assert st.regs[a.T2] == sext32(0x80000000)  # LO for negative dividend
    assert st.regs[a.T3] == sext32(-5)  # HI = dividend


def test_mthi_mtlo():
    st, mem = make()
    st.regs[a.T0] = 111
    st.regs[a.T1] = 222
    a.assemble(mem, CODE, a.mthi(a.T0), a.mtlo(a.T1), a.mflo(a.T2), a.mfhi(a.T3), a.NOP)
    for _ in range(4):
        step(st, mem)
    assert st.regs[a.T2] == 222
    assert st.regs[a.T3] == 111


# --- 64-bit integer semantics ---

def test_daddu_dsubu_full_width():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFF00000001
    st.regs[a.T1] = 0x0000000200000004
    a.assemble(
        mem, CODE,
        a.daddu(a.T2, a.T0, a.T1),
        a.dsubu(a.T3, a.T2, a.T1),
        a.NOP,
    )
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == 0x0000000100000005  # 64-bit wrap
    assert st.regs[a.T3] == sext64(0xFFFFFFFF00000001)


def test_daddiu_sign_extends_immediate():
    st, mem = make()
    st.regs[a.T0] = 0x8000000000000000
    a.assemble(mem, CODE, a.daddiu(a.T1, a.T0, -1),
               a.daddiu(a.T2, a.ZERO, -1), a.NOP)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T1] == 0x7FFFFFFFFFFFFFFF  # wraps at 64 bits
    assert st.regs[a.T2] == sext64(-1)


def test_64_bit_shifts():
    st, mem = make()
    st.regs[a.T0] = 0x8000000000000001
    a.assemble(
        mem, CODE,
        a.dsll(a.T1, a.T0, 4),
        a.dsrl(a.T2, a.T0, 4),
        a.dsra(a.T3, a.T0, 4),
        a.dsll32(a.T4, a.T0, 1),    # shift by 33
        a.dsrl32(a.T5, a.T0, 1),
        a.dsra32(a.T6, a.T0, 1),
        a.NOP,
    )
    for _ in range(6):
        step(st, mem)
    assert st.regs[a.T1] == 0x0000000000000010
    assert st.regs[a.T2] == 0x0800000000000000
    assert st.regs[a.T3] == sext64(0xF800000000000000)
    assert st.regs[a.T4] == 0x0000000200000000  # bit 63 shifted out, bit 0 -> 33
    assert st.regs[a.T5] == 0x0000000040000000  # bit 63 -> bit 30
    assert st.regs[a.T6] == sext64(0xFFFFFFFFC0000000)  # arithmetic: -2^30


def test_64_bit_variable_shifts_use_low_six_bits():
    st, mem = make()
    st.regs[a.T0] = 1
    st.regs[a.T1] = 65             # 65 & 0x3F == 1
    a.assemble(mem, CODE, a.dsllv(a.T2, a.T0, a.T1), a.NOP)
    step(st, mem)
    assert st.regs[a.T2] == 2


def test_slt_is_64_bit():
    st, mem = make()
    st.regs[a.T0] = 0xFFFFFFFFFFFFFFFF  # -1 in 64-bit
    st.regs[a.T1] = 1
    a.assemble(mem, CODE, a.slt(a.T2, a.T0, a.T1),
               a.sltu(a.T3, a.T0, a.T1), a.NOP)
    step(st, mem)
    step(st, mem)
    assert st.regs[a.T2] == 1
    assert st.regs[a.T3] == 0


def test_ld_sd_lwu_little_endian():
    st, mem = make()
    st.regs[a.T0] = STACK
    st.regs[a.T1] = 0x0123456789ABCDEF
    a.assemble(
        mem, CODE,
        a.sd(a.T1, 0, a.T0),
        a.ld(a.T2, 0, a.T0),
        a.lwu(a.T3, 4, a.T0),
        a.sw(a.T1, 8, a.T0),        # SW stores the low 32 bits only
        a.NOP,
    )
    for _ in range(4):
        step(st, mem)
    assert mem.read(STACK, 8) == b"\xEF\xCD\xAB\x89\x67\x45\x23\x01"
    assert mem.read(STACK + 8, 4) == b"\xEF\xCD\xAB\x89"
    assert st.regs[a.T2] == 0x0123456789ABCDEF
    assert st.regs[a.T3] == 0x01234567  # LE word at +4, zero-extended


def test_lq_sq_full_128_bits():
    st, mem = make()
    st.regs[a.T0] = STACK
    value = 0x0102030405060708090A0B0C0D0E0F10
    st.regs[a.T1] = value
    a.assemble(mem, CODE, a.sq(a.T1, 0, a.T0), a.lq(a.T2, 0, a.T0), a.NOP)
    step(st, mem)
    step(st, mem)
    assert mem.read(STACK, 16) == bytes(range(1, 17))[::-1]  # little-endian lanes
    assert st.regs[a.T2] == value
    assert st.regs[a.T2] > 0xFFFFFFFFFFFFFFFF  # the upper lane survived


def test_cop0_moves_and_interrupt_enable():
    st, mem = make()
    st.cop0[12] = 0x00400000
    st.regs[a.T0] = 0x00400002
    a.assemble(
        mem, CODE,
        a.enc_r(0x10, 0, a.T1, 12, 0, 0),   # mfc0 $t1, $12
        a.enc_r(0x10, 4, a.T0, 12, 0, 0),   # mtc0 $t0, $12
        a.NOP,
    )
    step(st, mem)
    assert st.regs[a.T1] == sext32(0x00400000)
    step(st, mem)
    assert st.cop0[12] == 0x00400002     # stored as a 32-bit register
    assert not st.cop0[12] & 1


def test_ei_di_and_eret():
    st, mem = make()
    st.cop0[14] = 0x00101234             # EPC
    a.assemble(
        mem, CODE,
        a.enc_r(0x10, 0x10, 0, 0, 0, 0x38),  # ei
        a.enc_r(0x10, 0x10, 0, 0, 0, 0x39),  # di
        a.enc_r(0x10, 0x10, 0, 0, 0, 0x18),  # eret
        a.NOP,
    )
    step(st, mem)
    assert st.cop0[12] & 1
    step(st, mem)
    assert not st.cop0[12] & 1
    step(st, mem)
    assert st.pc == 0x00101234


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
    st.regs[a.T0] = sext32(-1)       # bgezal not taken: link is not written
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
    assert mem.read(STACK, 4) == b"\xBE\xBA\xFE\xCA"

