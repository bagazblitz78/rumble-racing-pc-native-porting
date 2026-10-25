"""Directed differential tests: compiled native x64 must match the golden
interpreter bit-for-bit (regs, hi, lo, exit code) for every MVP instruction
and addressing edge case."""
from ps2toolkit.r5900.decoder import decode
from tests import asm as a
from tests.difftest import CODE, STACK, run_case


def _syscall_code(code: int):
    """syscall with an arbitrary 20-bit exit code."""
    return a.enc_r(0, 0, 0, (code >> 5) & 0x1F, code & 0x1F, 12)


def test_alu_arithmetic_and_logic():
    run_case([
        a.lui(a.T0, 0x1234),
        a.ori(a.T0, a.T0, 0x5678),
        a.addiu(a.T1, a.T0, 0x7F),
        a.addiu(a.T2, a.ZERO, -1),
        a.addu(a.T3, a.T0, a.T2),
        a.subu(a.T4, a.T0, a.T2),
        a.addi(a.T5, a.T0, -1000),
        a.and_(a.ZERO, a.T0, a.T1),   # write to $zero discarded on both sides
        a.and_(a.T7, a.T0, a.T2),
        a.or_(a.S0, a.T0, a.T2),
        a.xor_(a.S1, a.T0, a.T2),
        a.nor_(a.S2, a.T0, a.T2),
        a.slt(a.S3, a.T0, a.T2),
        a.sltu(a.S4, a.T0, a.T2),
        a.slti(a.S5, a.T0, -1),
        a.sltiu(a.S6, a.T0, 0xFFFF),
        a.andi(a.S7, a.T0, 0xFF),
        a.xori(a.T8, a.T0, 0xFFFF),
        a.ori(a.T9, a.ZERO, 0),       # rs == 0 immediate path
        a.NOP,
    ], label="alu")


def test_shifts():
    run_case([
        a.lui(a.T0, 0x8765),
        a.ori(a.T0, a.T0, 0x4321),
        a.sll(a.T1, a.T0, 4),
        a.srl(a.T2, a.T0, 8),
        a.sra(a.T3, a.T0, 31),
        a.sll(a.T4, a.T0, 0),
        a.addiu(a.T5, a.ZERO, 5),
        a.sllv(a.T6, a.T0, a.T5),
        a.srlv(a.T7, a.T0, a.T5),
        a.srav(a.S0, a.T0, a.T5),
        a.sllv(a.S1, a.T0, a.ZERO),  # shift by 0
        a.NOP,
    ], label="shifts")


def test_memory_word_and_byte():
    run_case([
        a.lui(a.T0, 0x1234),
        a.ori(a.T0, a.T0, 0x9ABC),
        a.sw(a.T0, 0, a.SP),
        a.lw(a.T1, 0, a.SP),
        a.lbu(a.T2, 0, a.SP),        # BE byte 0 = 0x12
        a.lb(a.T3, 1, a.SP),         # 0x23
        a.lbu(a.T4, 3, a.SP),        # 0x34
        a.lui(a.T6, 0x8000),
        a.sw(a.T6, 4, a.SP),
        a.lb(a.T7, 4, a.SP),         # sign-extended 0x80
        a.lh(a.S0, 0, a.SP),         # 0x1234
        a.lhu(a.S1, 4, a.SP),        # 0x8000
        a.lh(a.S2, 4, a.SP),         # 0xFFFF8000
        a.addiu(a.S3, a.ZERO, 0xAB),
        a.sb(a.S3, 8, a.SP),
        a.lbu(a.S4, 8, a.SP),
        a.addiu(a.S5, a.ZERO, -1),
        a.sh(a.S5, 10, a.SP),
        a.lhu(a.S6, 10, a.SP),       # 0xFFFF
        a.lw(a.S7, 8, a.SP),         # combined word
        a.NOP,
    ], label="memory")


def test_unaligned_pairs():
    run_case([
        a.lui(a.T0, 0xDEAD),
        a.ori(a.T0, a.T0, 0xBEEF),
        a.sw(a.T0, 0, a.SP),
        a.lui(a.T1, 0x0123),
        a.ori(a.T1, a.T1, 0x4567),
        a.sw(a.T1, 4, a.SP),
        a.addiu(a.T2, a.ZERO, 0xAAAA),
        a.lwl(a.T2, 1, a.SP),
        a.addiu(a.T3, a.ZERO, 0x5555),
        a.lwr(a.T3, 1, a.SP),
        a.lwl(a.T4, 2, a.SP),
        a.lwr(a.T4, 5, a.SP),
        a.lwl(a.T5, 3, a.SP),
        a.lwr(a.T5, 6, a.SP),
        a.lwl(a.T6, 0, a.SP),        # offset 0 edges
        a.lwr(a.T6, 3, a.SP),
        a.swl(a.T2, 1, a.SP),
        a.swr(a.T2, 4, a.SP),
        a.swl(a.T2, 2, a.SP),
        a.swr(a.T2, 5, a.SP),
        a.swl(a.T3, 3, a.SP),
        a.swr(a.T3, 6, a.SP),
        a.swl(a.T4, 0, a.SP),        # offset 0 edges
        a.swr(a.T4, 3, a.SP),
        a.lw(a.S0, 0, a.SP),
        a.lw(a.S1, 4, a.SP),
        a.lbu(a.S2, 1, a.SP),
        a.lbu(a.S3, 6, a.SP),
        a.NOP,
    ], label="unaligned")


def test_mult_div_including_quirks():
    run_case([
        a.lui(a.T0, 0x7FFF),
        a.ori(a.T0, a.T0, 0xFFFF),          # 0x7FFFFFFF
        a.lui(a.T1, 0x0002),
        a.ori(a.T1, a.T1, 0x0003),
        a.mult(a.T0, a.T1),
        a.mflo(a.T2),
        a.mfhi(a.T3),
        a.multu(a.T0, a.T1),
        a.mflo(a.T4),
        a.mfhi(a.T5),
        a.addiu(a.T6, a.ZERO, 7),
        a.div(a.T6, a.ZERO),                # div by zero
        a.mflo(a.S0),
        a.mfhi(a.S1),
        a.divu(a.T6, a.ZERO),
        a.mflo(a.S2),
        a.mfhi(a.S3),
        a.lui(a.S4, 0x8000),                # INT_MIN
        a.addiu(a.S5, a.ZERO, -1),
        a.div(a.S4, a.S5),                  # INT_MIN / -1
        a.mflo(a.S6),
        a.mfhi(a.S7),
        a.addiu(a.T8, a.ZERO, -7),
        a.addiu(a.T9, a.ZERO, 2),
        a.div(a.T8, a.T9),
        a.mflo(a.K0),
        a.mfhi(a.K1),
        a.divu(a.T8, a.T9),
        a.mflo(a.GP),
        a.addiu(a.V0, a.ZERO, 0x1234),
        a.mthi(a.V0),
        a.mtlo(a.V0),
        a.mfhi(a.V1),
        a.mflo(a.A0),
        a.NOP,
    ], label="multdiv")


def test_branches_taken_and_not_taken():
    run_case([
        a.addiu(a.T0, a.ZERO, 5),
        a.addiu(a.T1, a.ZERO, 5),
        a.beq(a.T0, a.T1, 2),               # taken
        a.addiu(a.T2, a.ZERO, 1),           # delay (executes)
        a.addiu(a.T3, a.ZERO, 99),          # skipped
        a.addiu(a.T4, a.ZERO, 2),           # target
        a.bne(a.T0, a.T1, 2),               # not taken
        a.addiu(a.T5, a.ZERO, 3),
        a.addiu(a.T6, a.ZERO, 4),
        a.blez(a.T0, 2),                    # not taken (5 > 0)
        a.addiu(a.T7, a.ZERO, 5),
        a.addiu(a.S0, a.ZERO, 6),
        a.bgtz(a.T0, 2),                    # taken
        a.addiu(a.S1, a.ZERO, 7),           # delay
        a.addiu(a.S2, a.ZERO, 99),          # skipped
        a.addiu(a.S3, a.ZERO, 8),
        a.addiu(a.S6, a.ZERO, -3),
        a.bgez(a.S6, 2),                    # not taken
        a.addiu(a.S4, a.ZERO, 9),
        a.addiu(a.S5, a.ZERO, 10),
        a.bltz(a.S6, 2),                    # taken
        a.addiu(a.S7, a.ZERO, 11),          # delay
        a.addiu(a.T8, a.ZERO, 99),          # skipped
        a.addiu(a.T9, a.ZERO, 12),
        a.NOP,
    ], label="branches")


def test_branch_condition_uses_predelay_state():
    # T0 == 0 at branch time; the delay slot clobbers T0 afterwards.
    run_case([
        a.addiu(a.T0, a.ZERO, 0),
        a.beq(a.T0, a.ZERO, 2),             # must see pre-delay T0 == 0
        a.addiu(a.T0, a.ZERO, 5),           # delay clobbers T0
        a.addiu(a.T1, a.ZERO, 99),          # skipped
        a.addiu(a.T2, a.ZERO, 1),
        a.bne(a.T0, a.ZERO, 2),             # T0 now 5 -> taken
        a.addiu(a.T0, a.ZERO, 0),           # delay clobbers again
        a.addiu(a.T3, a.ZERO, 99),          # skipped
        a.addiu(a.T4, a.ZERO, 2),
        a.NOP,
    ], label="predelay")


def test_loop_sum():
    run_case([
        a.addiu(a.T0, a.ZERO, 0),           # sum
        a.addiu(a.T1, a.ZERO, 1),           # i
        a.addiu(a.T2, a.ZERO, 11),          # limit
        a.addu(a.T0, a.T0, a.T1),           # body
        a.addiu(a.T1, a.T1, 1),
        a.bne(a.T1, a.T2, -3),              # -> body
        a.addiu(a.T3, a.ZERO, 0),           # delay
        a.NOP,
    ], label="loop")


def test_jal_function_call():
    run_case([
        a.sw(a.RA, 0, a.SP),                # save sentinel ra
        a.addiu(a.T0, a.ZERO, 1),
        a.jal(CODE + 0x40),
        a.sw(a.RA, 4, a.SP),                # delay: link already written?
        a.addu(a.T2, a.V0, a.ZERO),         # callee result in v0
        a.lw(a.RA, 0, a.SP),                # restore sentinel
        a.lw(a.T4, 4, a.SP),                # link value observed by delay slot
        a.jr(a.RA),
        a.NOP,
        a.NOP, a.NOP, a.NOP, a.NOP, a.NOP, a.NOP, a.NOP,  # pad to CODE+0x40
        a.lui(a.V0, 0xCAFE),                # @ CODE+0x40
        a.ori(a.V0, a.V0, 0x1234),
        a.addiu(a.A0, a.ZERO, 7),           # callee clobbers a0
        a.jr(a.RA),
        a.NOP,
    ], label="jal")


def test_syscall_exit_code():
    run_case([
        a.addiu(a.T0, a.ZERO, 5),
        a.addiu(a.T1, a.T0, 3),
        _syscall_code(42),
    ], label="syscall")


def test_kseg_address_translation():
    run_case([
        a.addiu(a.T0, a.ZERO, 0x77),
        a.lui(a.T1, 0xA011),                # 0xA0110000 == kseg1 mirror of stack
        a.sw(a.T0, 0, a.T1),
        a.lw(a.T2, 0, a.SP),
        a.lw(a.T3, 0, a.T1),
        a.lh(a.T4, 0, a.T1),
        a.sb(a.T0, 1, a.T1),
        a.lbu(a.T5, 1, a.SP),
        a.NOP,
    ], label="kseg")
