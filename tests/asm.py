"""Tiny EE assembler helpers for tests (field-level encodings)."""


def enc_r(op, rs, rt, rd, shamt, funct):
    return (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (shamt << 6) | funct


def enc_i(op, rs, rt, imm):
    return ((op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)) & 0xFFFFFFFF


def enc_j(op, target):
    return ((op << 26) | ((target >> 2) & 0x03FFFFFF)) & 0xFFFFFFFF


# register numbers
ZERO, AT, V0, V1, A0, A1, A2, A3 = range(8)
T0, T1, T2, T3, T4, T5, T6, T7 = range(8, 16)
S0, S1, S2, S3, S4, S5, S6, S7 = range(16, 24)
T8, T9, K0, K1, GP, SP, FP, RA = range(24, 32)


def addiu(rt, rs, imm):
    return enc_i(9, rs, rt, imm)


def addu(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 33)


def movz(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 10)


def movn(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 11)


def sync():
    return enc_r(0, 0, 0, 0, 0, 15)


def subu(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 35)


def dadd(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 44)


def daddu(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 45)


def dsub(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 46)


def dsubu(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 47)


def daddiu(rt, rs, imm):
    return enc_i(25, rs, rt, imm)


def daddi(rt, rs, imm):
    return enc_i(24, rs, rt, imm)


def or_(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 37)


def and_(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 36)


def xor_(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 38)


def nor_(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 39)


def slt(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 42)


def sltu(rd, rs, rt):
    return enc_r(0, rs, rt, rd, 0, 43)


def slti(rt, rs, imm):
    return enc_i(10, rs, rt, imm)


def sltiu(rt, rs, imm):
    return enc_i(11, rs, rt, imm)


def addi(rt, rs, imm):
    return enc_i(8, rs, rt, imm)


def andi(rt, rs, imm):
    return enc_i(12, rs, rt, imm)


def ori(rt, rs, imm):
    return enc_i(13, rs, rt, imm)


def xori(rt, rs, imm):
    return enc_i(14, rs, rt, imm)


def lui(rt, imm):
    return enc_i(15, 0, rt, imm)


def sll(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 0)


def srl(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 2)


def sra(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 3)


def sllv(rd, rt, rs):
    return enc_r(0, rs, rt, rd, 0, 4)


def srlv(rd, rt, rs):
    return enc_r(0, rs, rt, rd, 0, 6)


def srav(rd, rt, rs):
    return enc_r(0, rs, rt, rd, 0, 7)


def dsll(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 56)


def dsrl(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 58)


def dsra(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 59)


def dsll32(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 60)


def dsrl32(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 62)


def dsra32(rd, rt, shamt):
    return enc_r(0, 0, rt, rd, shamt, 63)


def dsllv(rd, rt, rs):
    return enc_r(0, rs, rt, rd, 0, 20)


def dsrlv(rd, rt, rs):
    return enc_r(0, rs, rt, rd, 0, 22)


def dsrav(rd, rt, rs):
    return enc_r(0, rs, rt, rd, 0, 23)


def jr(rs):
    return enc_r(0, rs, 0, 0, 0, 8)


def jalr(rd, rs):
    return enc_r(0, rs, 0, rd, 0, 9)


def jal(target):
    return enc_j(3, target)


def j(target):
    return enc_j(2, target)


def beq(rs, rt, off):
    return enc_i(4, rs, rt, off)


def bne(rs, rt, off):
    return enc_i(5, rs, rt, off)


def blez(rs, off):
    return enc_i(6, rs, 0, off)


def bgtz(rs, off):
    return enc_i(7, rs, 0, off)


def bltz(rs, off):
    return enc_i(1, rs, 0, off)


def bgez(rs, off):
    return enc_i(1, rs, 1, off)


def beql(rs, rt, off):
    return enc_i(20, rs, rt, off)


def bnel(rs, rt, off):
    return enc_i(21, rs, rt, off)


def blezl(rs, off):
    return enc_i(22, rs, 0, off)


def bgtzl(rs, off):
    return enc_i(23, rs, 0, off)


def _regimm(rs, rt_field, off):
    return enc_i(1, rs, rt_field, off)


def bltzl(rs, off):
    return _regimm(rs, 2, off)


def bgezl(rs, off):
    return _regimm(rs, 3, off)


def bltzal(rs, off):
    return _regimm(rs, 16, off)


def bgezal(rs, off):
    return _regimm(rs, 17, off)


def bltzall(rs, off):
    return _regimm(rs, 18, off)


def bgezall(rs, off):
    return _regimm(rs, 19, off)


def lb(rt, off, base):
    return enc_i(32, base, rt, off)


def lh(rt, off, base):
    return enc_i(33, base, rt, off)


def lwl(rt, off, base):
    return enc_i(34, base, rt, off)


def lw(rt, off, base):
    return enc_i(35, base, rt, off)


def lbu(rt, off, base):
    return enc_i(36, base, rt, off)


def lhu(rt, off, base):
    return enc_i(37, base, rt, off)


def lwu(rt, off, base):
    return enc_i(39, base, rt, off)


def ld(rt, off, base):
    return enc_i(55, base, rt, off)


def sd(rt, off, base):
    return enc_i(63, base, rt, off)


def lwr(rt, off, base):
    return enc_i(38, base, rt, off)


def sb(rt, off, base):
    return enc_i(40, base, rt, off)


def sh(rt, off, base):
    return enc_i(41, base, rt, off)


def swl(rt, off, base):
    return enc_i(42, base, rt, off)


def sw(rt, off, base):
    return enc_i(43, base, rt, off)


def swr(rt, off, base):
    return enc_i(46, base, rt, off)


def cache(op, off, base):
    return enc_i(47, base, op, off)


def pref(op, off, base):
    return enc_i(51, base, op, off)


def mfhi(rd):
    return enc_r(0, 0, 0, rd, 0, 16)


def mthi(rs):
    return enc_r(0, rs, 0, 0, 0, 17)


def mflo(rd):
    return enc_r(0, 0, 0, rd, 0, 18)


def mtlo(rs):
    return enc_r(0, rs, 0, 0, 0, 19)


def mult(rs, rt):
    return enc_r(0, rs, rt, 0, 0, 24)


def multu(rs, rt):
    return enc_r(0, rs, rt, 0, 0, 25)


def div(rs, rt):
    return enc_r(0, rs, rt, 0, 0, 26)


def divu(rs, rt):
    return enc_r(0, rs, rt, 0, 0, 27)


def syscall():
    return enc_r(0, 0, 0, 0, 0, 12)


NOP = 0


def assemble(mem, addr, *words):
    """Write instruction words into guest memory starting at addr."""
    data = b"".join((w & 0xFFFFFFFF).to_bytes(4, "big") for w in words)
    mem.write(addr, data)
    return addr
