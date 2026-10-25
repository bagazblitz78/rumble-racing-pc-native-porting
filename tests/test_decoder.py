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
