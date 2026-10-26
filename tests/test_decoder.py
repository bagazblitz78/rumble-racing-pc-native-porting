"""Tests for the R5900 instruction decoder (T2.2).

Encodings are built field-by-field from the MIPS instruction formats so
tests are independent of the implementation.
"""
import pytest

from ps2toolkit.r5900.decoder import DecodedInsn, decode

REG_NAMES = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra",
]


def enc_r(op, rs, rt, rd, shamt, funct):
    return (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (shamt << 6) | funct


def enc_i(op, rs, rt, imm):
    return ((op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)) & 0xFFFFFFFF


def enc_j(op, target):
    return ((op << 26) | ((target >> 2) & 0x03FFFFFF)) & 0xFFFFFFFF


ADDR = 0x00101000


def d(word):
    return decode(word, ADDR)


def test_addiu():
    insn = d(enc_i(9, 4, 2, 0x20))  # addiu $v0, $a0, 0x20
    assert insn.mnemonic == "addiu"
    assert insn.rt == 2 and insn.rs == 4
    assert insn.imm == 0x20


def test_addiu_negative_immediate_sign_extends():
    insn = d(enc_i(9, 4, 2, 0xFFF0))
    assert insn.imm == -16


def test_addu():
    insn = d(enc_r(0, 8, 9, 10, 0, 33))  # addu $t2, $t0, $t1
    assert insn.mnemonic == "addu"
    assert (insn.rs, insn.rt, insn.rd) == (8, 9, 10)


def test_or_and_lui():
    assert d(enc_r(0, 8, 9, 10, 0, 37)).mnemonic == "or"
    assert d(enc_r(0, 8, 9, 10, 0, 36)).mnemonic == "and"
    insn = d(enc_i(15, 0, 8, 0xABCD))
    assert insn.mnemonic == "lui"
    assert insn.rt == 8 and insn.uimm == 0xABCD  # lui uses zero-extended imm


def test_shifts():
    assert d(enc_r(0, 0, 8, 9, 4, 0)).mnemonic == "sll"
    assert d(enc_r(0, 0, 8, 9, 4, 2)).mnemonic == "srl"
    assert d(enc_r(0, 0, 8, 9, 4, 3)).mnemonic == "sra"
    sllv = d(enc_r(0, 10, 8, 9, 0, 4))
    assert sllv.mnemonic == "sllv"
    assert (sllv.rs, sllv.rt, sllv.rd) == (10, 8, 9)


def test_branches():
    beq = d(enc_i(4, 8, 9, 4))
    assert beq.mnemonic == "beq"
    assert beq.imm == 4  # word offset
    bne = d(enc_i(5, 8, 9, -2))
    assert bne.imm == -2
    assert d(enc_i(6, 8, 0, 4)).mnemonic == "blez"
    assert d(enc_i(7, 8, 0, 4)).mnemonic == "bgtz"
    bltz = d((1 << 26) | (8 << 21) | (0 << 16) | 4)
    assert bltz.mnemonic == "bltz"
    bgez = d((1 << 26) | (8 << 21) | (1 << 16) | 4)
    assert bgez.mnemonic == "bgez"


def test_branch_likely_and_link_forms():
    beql = d(enc_i(20, 8, 9, 3))
    assert beql.mnemonic == "beql" and beql.is_branch and beql.is_likely
    bnel = d(enc_i(21, 8, 9, 3))
    assert bnel.mnemonic == "bnel" and bnel.is_likely
    assert d(enc_i(22, 8, 0, 3)).mnemonic == "blezl"
    assert d(enc_i(23, 8, 0, 3)).mnemonic == "bgtzl"
    bgezal = d((1 << 26) | (8 << 21) | (17 << 16) | 3)
    assert bgezal.mnemonic == "bgezal"
    assert bgezal.is_branch_link and bgezal.is_branch
    assert d((1 << 26) | (8 << 21) | (2 << 16) | 3).mnemonic == "bltzl"
    assert d((1 << 26) | (8 << 21) | (18 << 16) | 3).mnemonic == "bltzall"


def test_movz_movn_sync():
    movz = d(enc_r(0, 8, 9, 10, 0, 10))
    assert movz.mnemonic == "movz"
    assert (movz.rs, movz.rt, movz.rd) == (8, 9, 10)
    assert d(enc_r(0, 8, 9, 10, 0, 11)).mnemonic == "movn"
    assert d(enc_r(0, 0, 0, 0, 0, 15)).mnemonic == "sync"


def test_cache_and_pref():
    cache = d(enc_i(47, 8, 3, 4))
    assert cache.mnemonic == "cache"
    assert (cache.rs, cache.rt, cache.imm) == (8, 3, 4)
    pref = d(enc_i(51, 8, 2, 4))
    assert pref.mnemonic == "pref"
    assert (pref.rs, pref.rt) == (8, 2)


def test_jal_target():
    insn = d(enc_j(3, 0x00108000))
    assert insn.mnemonic == "jal"
    assert insn.target == 0x00108000


def test_jr_jalr():
    jr = d(enc_r(0, 31, 0, 0, 0, 8))
    assert jr.mnemonic == "jr"
    assert jr.rs == 31
    jalr = d(enc_r(0, 31, 0, 31, 0, 9))
    assert jalr.mnemonic == "jalr"
    assert jalr.rd == 31


def test_loads_stores():
    assert d(enc_i(32, 8, 9, 4)).mnemonic == "lb"
    assert d(enc_i(35, 8, 9, 4)).mnemonic == "lw"
    assert d(enc_i(36, 8, 9, 4)).mnemonic == "lbu"
    assert d(enc_i(43, 8, 9, 4)).mnemonic == "sw"
    assert d(enc_i(38, 8, 9, 4)).mnemonic == "lwr"  # 0x26
    assert d(enc_i(34, 8, 9, 4)).mnemonic == "lwl"  # 0x22
    assert d(enc_i(46, 8, 9, 4)).mnemonic == "swr"  # 0x2E
    assert d(enc_i(42, 8, 9, 4)).mnemonic == "swl"  # 0x2A


def test_64_bit_integer_forms():
    daddu = d(enc_r(0, 8, 9, 10, 0, 45))
    assert daddu.mnemonic == "daddu"
    assert (daddu.rs, daddu.rt, daddu.rd) == (8, 9, 10)
    assert d(enc_r(0, 8, 9, 10, 0, 44)).mnemonic == "dadd"
    assert d(enc_r(0, 8, 9, 10, 0, 46)).mnemonic == "dsub"
    assert d(enc_r(0, 8, 9, 10, 0, 47)).mnemonic == "dsubu"
    assert d(enc_i(25, 8, 10, -4)).mnemonic == "daddiu"
    assert d(enc_i(24, 8, 10, -4)).mnemonic == "daddi"

    dsll = d(enc_r(0, 0, 8, 9, 4, 56))
    assert dsll.mnemonic == "dsll" and dsll.shamt == 4
    assert d(enc_r(0, 0, 8, 9, 4, 58)).mnemonic == "dsrl"
    assert d(enc_r(0, 0, 8, 9, 4, 59)).mnemonic == "dsra"
    assert d(enc_r(0, 0, 8, 9, 4, 60)).mnemonic == "dsll32"
    assert d(enc_r(0, 0, 8, 9, 4, 62)).mnemonic == "dsrl32"
    assert d(enc_r(0, 0, 8, 9, 4, 63)).mnemonic == "dsra32"
    assert d(enc_r(0, 10, 8, 9, 0, 20)).mnemonic == "dsllv"
    assert d(enc_r(0, 10, 8, 9, 0, 22)).mnemonic == "dsrlv"
    assert d(enc_r(0, 10, 8, 9, 0, 23)).mnemonic == "dsrav"


def test_64_bit_memory_forms():
    ld = d(enc_i(55, 8, 9, 16))
    assert ld.mnemonic == "ld" and ld.is_load
    sd = d(enc_i(63, 8, 9, 16))
    assert sd.mnemonic == "sd" and sd.is_store
    lwu = d(enc_i(39, 8, 9, 16))
    assert lwu.mnemonic == "lwu" and lwu.is_load


def test_lq_sq():
    lq = d(enc_i(0x1E, 8, 9, 16))
    assert lq.mnemonic == "lq" and lq.is_load
    assert (lq.rt, lq.rs, lq.imm) == (9, 8, 16)
    sq = d(enc_i(0x1F, 8, 9, 48))
    assert sq.mnemonic == "sq" and sq.is_store
    assert (sq.rt, sq.rs, sq.imm) == (9, 8, 48)


def test_cop0_moves_and_control():
    mfc0 = d(enc_r(0x10, 0, 9, 12, 0, 0))
    assert mfc0.mnemonic == "mfc0"
    assert (mfc0.rt, mfc0.rd) == (9, 12)
    mtc0 = d(enc_r(0x10, 4, 9, 12, 0, 0))
    assert mtc0.mnemonic == "mtc0"
    assert (mtc0.rt, mtc0.rd) == (9, 12)
    assert d((0x10 << 26) | (0x10 << 21) | 0x38).mnemonic == "ei"
    assert d((0x10 << 26) | (0x10 << 21) | 0x39).mnemonic == "di"
    assert d((0x10 << 26) | (0x10 << 21) | 0x18).mnemonic == "eret"


def test_mult_div_mfhi_mflo():
    assert d(enc_r(0, 8, 9, 0, 0, 24)).mnemonic == "mult"
    assert d(enc_r(0, 8, 9, 0, 0, 25)).mnemonic == "multu"
    assert d(enc_r(0, 8, 9, 0, 0, 26)).mnemonic == "div"
    assert d(enc_r(0, 8, 9, 0, 0, 27)).mnemonic == "divu"
    mfhi = d(enc_r(0, 0, 0, 8, 0, 16))
    assert mfhi.mnemonic == "mfhi" and mfhi.rd == 8
    mtlo = d(enc_r(0, 8, 0, 0, 0, 19))
    assert mtlo.mnemonic == "mtlo" and mtlo.rs == 8


def test_syscall_and_nop():
    assert d(enc_r(0, 0, 0, 0, 0, 12)).mnemonic == "syscall"
    nop = d(0)
    assert nop.mnemonic == "nop"


def test_unsupported_reports_opcode():
    insn = d(enc_r(0, 8, 9, 10, 0, 52))  # (unused funct in subset)
    assert insn.mnemonic == "unsupported"
    insn2 = d(enc_i(28, 8, 9, 0))  # MMI opcode family - not in MVP subset
    assert insn2.mnemonic == "unsupported"
    assert insn2.opcode == 28


def test_slt_sltu():
    assert d(enc_r(0, 8, 9, 10, 0, 42)).mnemonic == "slt"
    assert d(enc_r(0, 8, 9, 10, 0, 43)).mnemonic == "sltu"


def test_disasm_line():
    insn: DecodedInsn = d(enc_i(9, 4, 2, 0x20))
    text = insn.disasm()
    assert "addiu" in text
    assert "$v0" in text
    assert "0x20" in text
