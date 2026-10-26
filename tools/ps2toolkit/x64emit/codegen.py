"""x64 code generation from recompiled EE CFGs (task T2.7 core).

Generated-code calling convention (self-contained, closed world):
  - one native function `int run(void* ctx)` (Win64: ctx in rcx)
  - ctx layout (offsets): regs[i] = i*16 (128-bit: low u64 at +0, high u64
    at +8), hi = 512, lo = 528 (128-bit each), guest_ram_ptr = 544,
    fpr[32] = 552 (raw float bits), fcr31 = 680, fpu_acc = 684; guest RAM
    buffer starts at 0x2B0
  - r15 = ctx pointer, r14 = guest RAM base, r10 = branch-condition scratch;
    rax/rcx/rdx/r8/r9 are integer scratch, xmm0-xmm3 are SIMD scratch
  - returns -1 (0xFFFFFFFF) on normal completion, else the syscall code
  - generated code never calls OS APIs directly. It uses unaligned SSE2
    loads/stores and no stack slots, so no ABI shadow-space dance is needed
    between generated functions (documented deviation from Win64 ABI; safe
    because only generated code is ever called).

Semantics mirror tools/ps2toolkit/golden/interpreter.py exactly,
including PS2 DIV quirks and little-endian LWL/LWR/SWL/SWR pair rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..golden.interpreter import UnsupportedInstruction
from ..ir.cfg import (
    BranchTerm,
    CallTerm,
    FallTo,
    Function,
    IndirectTerm,
    JumpTerm,
    ReturnTerm,
    StopTerm,
    build_function,
)
from ..r5900.decoder import DecodedInsn
from .encoder import R8, R9, R10, R14, R15, RAX, RCX, RDX, Asm, Label

M32 = 0xFFFFFFFF
HI_OFF = 512
LO_OFF = 528
RAM_PTR_OFF = 544
FPR_OFF = 552
FCR31_OFF = 680
ACC_OFF = 684
CTX_SIZE = 688
RAM_OFF = RAM_PTR_OFF
REG_SIZE = 16
EXIT_NORMAL = 0xFFFFFFFF  # -1

_SHIFT = {"sll": "shl", "srl": "shr", "sra": "sar",
          "sllv": "shl", "srlv": "shr", "srav": "sar",
          "dsll": "shl", "dsrl": "shr", "dsra": "sar",
          "dsll32": "shl", "dsrl32": "shr", "dsra32": "sar",
          "dsllv": "shl", "dsrlv": "shr", "dsrav": "sar"}
_BRANCH_CC = {
    "blez": "le", "blezl": "le",
    "bgtz": "g", "bgtzl": "g",
    "bltz": "l", "bltzl": "l", "bltzal": "l", "bltzall": "l",
    "bgez": "ge", "bgezl": "ge", "bgezal": "ge", "bgezall": "ge",
}


class CodeGenError(Exception):
    """Raised when a program cannot be translated to native x64."""


@dataclass
class Program:
    code: bytes
    entry_offset: int
    function_offsets: dict[int, int] = field(default_factory=dict)


def _slot(asm: Asm, dst: int, reg: int):
    """ld32 dst <- low 32 bits of GPR[reg] (zero-extended)."""
    asm.ld32(dst, R15, reg * 16)


def _slot64(asm: Asm, dst: int, reg: int):
    """ld64 dst <- low 64 bits of GPR[reg]."""
    asm.ld64_rm(dst, R15, reg * 16)


def _store_high_sign(asm: Asm, reg: int, src: int):
    """Write the upper 64 bits as the sign extension of `src`'s bit 63."""
    asm.mov_rr(RDX, src)
    asm.shift_ri64("sar", RDX, 63)
    asm.st64(R15, reg * 16 + 8, RDX)


def _sto(asm: Asm, reg: int, src: int):
    """GPR[reg] <- sign-extended 32-bit src ($zero writes discarded)."""
    if reg == 0:
        return
    asm.movsxd(src, src)
    asm.st64(R15, reg * 16, src)
    _store_high_sign(asm, reg, src)


def _sto64(asm: Asm, reg: int, src: int):
    """GPR[reg] <- sign-extended 64-bit src ($zero writes discarded)."""
    if reg == 0:
        return
    asm.st64(R15, reg * 16, src)
    _store_high_sign(asm, reg, src)


def _xmm_slot(asm: Asm, xmm: int, reg: int):
    asm.movdqu_ld(xmm, R15, reg * 16)


def _xmm_store(asm: Asm, reg: int, xmm: int):
    if reg != 0:
        asm.movdqu_st(R15, reg * 16, xmm)


def _addr32(asm: Asm, base_reg: int, imm: int, dst: int):
    """dst <- (GPR[base_reg] + imm) in RCX, masked to physical."""
    asm.ld32(dst, R15, base_reg * 16)
    asm.alu_ri("add", dst, imm)
    asm.mov_rr(RCX, dst)
    asm.alu_ri("and", RCX, 0x1FFFFFFF)


def emit_insn(asm: Asm, insn: DecodedInsn, sys_exit: Label | None = None):
    """Emit one non-control-flow guest instruction."""
    m = insn.mnemonic
    rs, rt, rd = insn.rs, insn.rt, insn.rd

    if m == "nop":
        return
    if m in ("addiu", "addi"):
        if rs == 0:
            asm.mov_ri(RAX, insn.imm & M32)
        else:
            _slot(asm, RAX, rs)
            asm.alu_ri("add", RAX, insn.imm)
        _sto(asm, rt, RAX)
    elif m in ("daddiu", "daddi"):
        _slot64(asm, RAX, rs)
        asm.alu_ri64("add", RAX, insn.imm)
        _sto64(asm, rt, RAX)
    elif m == "andi":
        # 128-bit logical AND with a zero-extended immediate: upper = 0
        _slot64(asm, RAX, rs)
        asm.alu_ri64("and", RAX, insn.uimm)
        if rt:
            asm.st64(R15, rt * 16, RAX)
            asm.mov_ri(RDX, 0)
            asm.st64(R15, rt * 16 + 8, RDX)
    elif m in ("ori", "xori"):
        # 128-bit logical OR/XOR: only the low 64 bits change, but the
        # destination's upper half must come from rs (it is not preserved
        # when rd != rs).
        _slot64(asm, RAX, rs)
        asm.alu_ri64("or" if m == "ori" else "xor", RAX, insn.uimm)
        if rt:
            asm.st64(R15, rt * 16, RAX)
            asm.ld64_rm(RCX, R15, rs * 16 + 8)
            asm.st64(R15, rt * 16 + 8, RCX)
    elif m == "lui":
        asm.mov_ri(RAX, (insn.uimm << 16) & M32)
        _sto(asm, rt, RAX)
    elif m in ("slti", "sltiu"):
        _slot64(asm, RAX, rs)
        asm.alu_ri64("cmp", RAX, insn.imm)
        asm.setcc("l" if m == "slti" else "b", RDX)
        asm.movzx(RDX, RDX)
        _sto(asm, rt, RDX)
    elif m in ("sll", "srl", "sra"):
        _slot(asm, RAX, rt)
        asm.shift_ri(_SHIFT[m], RAX, insn.shamt)
        _sto(asm, rd, RAX)
    elif m in ("sllv", "srlv", "srav"):
        _slot(asm, RAX, rt)
        _slot(asm, RCX, rs)
        asm.shift_cl(_SHIFT[m], RAX)
        _sto(asm, rd, RAX)
    elif m in ("dsll", "dsrl", "dsra", "dsll32", "dsrl32", "dsra32"):
        amount = insn.shamt + (32 if m.endswith("32") else 0)
        _slot64(asm, RAX, rt)
        asm.shift_ri64(_SHIFT[m], RAX, amount)
        _sto64(asm, rd, RAX)
    elif m in ("dsllv", "dsrlv", "dsrav"):
        _slot64(asm, RAX, rt)
        _slot(asm, RCX, rs)  # x86 masks the count to 6 bits, like MIPS64
        asm.shift_cl64(_SHIFT[m], RAX)
        _sto64(asm, rd, RAX)
    elif m in ("add", "addu"):
        _slot(asm, RAX, rs)
        _slot(asm, RCX, rt)
        asm.alu_rr("add", RAX, RCX)
        _sto(asm, rd, RAX)
    elif m in ("sub", "subu"):
        _slot(asm, RAX, rs)
        _slot(asm, RCX, rt)
        asm.alu_rr("sub", RAX, RCX)
        _sto(asm, rd, RAX)
    elif m in ("dadd", "daddu", "dsub", "dsubu"):
        _slot64(asm, RAX, rs)
        _slot64(asm, RCX, rt)
        asm.alu_rr64("add" if m in ("dadd", "daddu") else "sub", RAX, RCX)
        _sto64(asm, rd, RAX)
    elif m in ("and", "or", "xor"):
        _xmm_slot(asm, 0, rs)
        _xmm_slot(asm, 1, rt)
        {"and": asm.pand, "or": asm.por, "xor": asm.pxor}[m](0, 1)
        _xmm_store(asm, rd, 0)
    elif m == "nor":
        _xmm_slot(asm, 0, rs)
        _xmm_slot(asm, 1, rt)
        asm.por(0, 1)
        asm.pcmpeqd(2, 2)
        asm.pxor(0, 2)
        _xmm_store(asm, rd, 0)
    elif m in ("movz", "movn"):
        # 128-bit conditional move; the condition tests the whole register
        _slot64(asm, RAX, rt)
        asm.ld64_rm(RCX, R15, rt * 16 + 8)
        asm.alu_rr64("or", RAX, RCX)
        skip = asm.new_label()
        asm.test_rr64(RAX, RAX)
        asm.jcc("ne" if m == "movz" else "e", skip)
        _xmm_slot(asm, 0, rs)
        _xmm_store(asm, rd, 0)
        asm.bind(skip)
    elif m in ("sync", "cache", "pref"):
        return  # no cache/coherence model in the single-threaded recompiler
    elif insn.opcode == 0x1C:
        emit_mmi(asm, insn)
    elif insn.opcode == 0x11 or m in ("lwc1", "swc1"):
        emit_cop1(asm, insn)
    elif m in ("slt", "sltu"):
        _slot64(asm, RAX, rs)
        _slot64(asm, RCX, rt)
        asm.alu_rr64("cmp", RAX, RCX)
        asm.setcc("l" if m == "slt" else "b", RDX)
        asm.movzx(RDX, RDX)
        _sto(asm, rd, RDX)
    elif m == "mult":  # signed: 64-bit product via sign-extended imul
        _slot(asm, RAX, rs)
        asm.movsxd(RAX, RAX)
        _slot(asm, RCX, rt)
        asm.movsxd(RCX, RCX)
        asm.imul64_rr(RAX, RCX)
        asm.mov_rr(RDX, RAX)
        asm.shr64_ri(RDX, 32)  # NOT shift_ri: x86 masks 32-bit counts mod 32
        asm.st32(R15, HI_OFF, RDX)
        asm.st32(R15, LO_OFF, RAX)
    elif m == "multu":
        _slot(asm, RAX, rs)
        _slot(asm, RCX, rt)
        asm.mul_r(RCX)  # rdx:rax = 64-bit product
        asm.st32(R15, HI_OFF, RDX)
        asm.st32(R15, LO_OFF, RAX)
    elif m == "div":
        _slot(asm, RAX, rs)  # eax = dividend
        _slot(asm, RCX, rt)  # ecx = divisor
        jz_zero, jne_norm, jns, done = (asm.new_label() for _ in range(4))
        asm.test_rr(RCX, RCX)
        asm.jcc("z", jz_zero)
        asm.alu_ri("cmp", RCX, -1)  # overflow only if divisor == -1 ...
        asm.jcc("ne", jne_norm)
        asm.alu_ri("cmp", RAX, 0x80000000)  # ... and dividend == INT_MIN
        asm.jcc("ne", jne_norm)
        asm.mov_ri(RAX, 0x80000000)  # INT_MIN / -1
        asm.mov_ri(RDX, 0)
        asm.st32(R15, LO_OFF, RAX)
        asm.st32(R15, HI_OFF, RDX)
        asm.jmp(done)
        asm.bind(jne_norm)
        asm.cdq()
        asm.idiv_r(RCX)
        asm.st32(R15, LO_OFF, RAX)
        asm.st32(R15, HI_OFF, RDX)
        asm.jmp(done)
        asm.bind(jz_zero)
        asm.mov_ri(RDX, 0x7FFFFFFF)
        asm.test_rr(RAX, RAX)
        asm.jcc("ns", jns)
        asm.mov_ri(RDX, 0x80000000)
        asm.bind(jns)
        asm.st32(R15, LO_OFF, RDX)
        asm.st32(R15, HI_OFF, RAX)  # HI = dividend
        asm.bind(done)
    elif m == "divu":
        _slot(asm, RAX, rs)
        _slot(asm, RCX, rt)
        jz, done = asm.new_label(), asm.new_label()
        asm.test_rr(RCX, RCX)
        asm.jcc("z", jz)
        asm.mov_rr(R8, RCX)
        asm.mov_ri(RDX, 0)
        asm.div_r(R8)
        asm.st32(R15, LO_OFF, RAX)
        asm.st32(R15, HI_OFF, RDX)
        asm.jmp(done)
        asm.bind(jz)
        asm.mov_ri(RDX, M32)
        asm.st32(R15, LO_OFF, RDX)
        asm.st32(R15, HI_OFF, RAX)
        asm.bind(done)
    elif m == "mfhi":
        asm.ld32(RAX, R15, HI_OFF)
        _sto(asm, rd, RAX)
    elif m == "mflo":
        asm.ld32(RAX, R15, LO_OFF)
        _sto(asm, rd, RAX)
    elif m == "mthi":
        _slot(asm, RAX, rs)
        asm.st32(R15, HI_OFF, RAX)
    elif m == "mtlo":
        _slot(asm, RAX, rs)
        asm.st32(R15, LO_OFF, RAX)
    elif m == "syscall":
        if sys_exit is None:
            raise CodeGenError(f"syscall outside main at {insn.addr:#010x}")
        asm.mov_ri(RAX, (insn.raw >> 6) & 0xFFFFF)
        asm.jmp(sys_exit)
    elif m == "break":
        raise CodeGenError(f"break unsupported at {insn.addr:#010x}")
    elif m == "lb":
        _slot(asm, RAX, rs)
        asm.alu_ri("add", RAX, insn.imm)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 0x1FFFFFFF)
        asm.ld8_ram(RDX, R14, RCX)
        asm.movzx(RAX, RDX)
        asm.movsx8(RAX, RAX)
        _sto(asm, rt, RAX)
    elif m == "lbu":
        _slot(asm, RAX, rs)
        asm.alu_ri("add", RAX, insn.imm)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 0x1FFFFFFF)
        asm.ld8_ram(RDX, R14, RCX)
        asm.movzx(RAX, RDX)
        _sto(asm, rt, RAX)
    elif m == "lh":
        _slot(asm, RAX, rs)
        asm.alu_ri("add", RAX, insn.imm)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 0x1FFFFFFF)
        asm.ld16_ram(RDX, R14, RCX)
        asm.movsx16(RAX, RDX)
        _sto(asm, rt, RAX)
    elif m == "lhu":
        _slot(asm, RAX, rs)
        asm.alu_ri("add", RAX, insn.imm)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 0x1FFFFFFF)
        asm.ld16_ram(RDX, R14, RCX)
        asm.movzx16(RAX, RDX)
        _sto(asm, rt, RAX)
    elif m == "lw":
        _addr32(asm, rs, insn.imm, RAX)
        asm.ld32_ram(RAX, R14, RCX)
        _sto(asm, rt, RAX)
    elif m == "lwu":
        _addr32(asm, rs, insn.imm, RAX)
        asm.ld32_ram(RAX, R14, RCX)
        _sto64(asm, rt, RAX)
    elif m == "ld":
        _addr32(asm, rs, insn.imm, RAX)
        asm.ld64_ram(RAX, R14, RCX)
        _sto64(asm, rt, RAX)
    elif m == "lq":
        _addr32(asm, rs, insn.imm, RAX)
        asm.movdqu_ld_idx(0, R14, RCX)
        _xmm_store(asm, rt, 0)
    elif m == "sq":
        _xmm_slot(asm, 0, rt)
        _addr32(asm, rs, insn.imm, RAX)
        asm.movdqu_st_idx(R14, RCX, 0)
    elif m == "sb":
        _slot(asm, RDX, rt)
        _addr32(asm, rs, insn.imm, RAX)
        asm.st8_ram(R14, RCX, RDX)
    elif m == "sh":
        _slot(asm, RAX, rt)
        _addr32(asm, rs, insn.imm, RDX)
        asm.st16_ram(R14, RCX, RAX)
    elif m == "sw":
        _slot(asm, RDX, rt)
        _addr32(asm, rs, insn.imm, RAX)
        asm.st32_ram(R14, RCX, RDX)
    elif m == "sd":
        _slot64(asm, RDX, rt)
        _addr32(asm, rs, insn.imm, RAX)
        asm.st64_ram(R14, RCX, RDX)
    elif m == "lwl":
        # LE LWL: load W's high bytes into rt's high bytes, keep rt's low s bits
        asm.ld32(R8, R15, rt * 16)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(RDX, RCX)
        asm.alu_ri("and", RDX, 3)
        asm.shift_ri("shl", RDX, 3)         # s = 8*(addr&3)
        asm.alu_ri("and", RCX, 0xFFFFFFFC)
        asm.ld32_ram(RAX, R14, RCX)         # W
        asm.mov_rr(RCX, RDX)
        asm.mov_ri(R9, 1)
        asm.shift_cl("shl", R9)             # 1 << s
        asm.alu_ri("sub", R9, 1)            # mask = low s bits
        asm.mov_rr(RCX, R9)
        asm.not_r(RCX)
        asm.alu_rr("and", RAX, RCX)         # W & ~mask
        asm.alu_rr("and", R8, R9)           # rt & mask
        asm.alu_rr("or", RAX, R8)
        _sto(asm, rt, RAX)
    elif m == "lwr":
        # LE LWR: load W's low n bytes into rt's low bytes, keep rt's high bits
        asm.ld32(R8, R15, rt * 16)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(RDX, RCX)
        asm.alu_ri("and", RDX, 3)
        asm.shift_ri("shl", RDX, 3)         # s
        asm.alu_ri("and", RCX, 0xFFFFFFFC)
        asm.ld32_ram(RAX, R14, RCX)         # W
        asm.mov_ri(RCX, 24)
        asm.alu_rr("sub", RCX, RDX)         # 24 - s
        asm.mov_ri(R9, M32)
        asm.shift_cl("shr", R9)             # mask = (1 << 8n) - 1
        asm.mov_rr(RCX, R9)
        asm.not_r(RCX)
        asm.alu_rr("and", R8, RCX)          # rt & ~mask
        asm.alu_rr("and", RAX, R9)          # W & mask
        asm.alu_rr("or", RAX, R8)
        _sto(asm, rt, RAX)
    elif m == "swl":
        # LE SWL: store rt's high bytes over W's high bytes at addr..word_end
        asm.ld32(RDX, R15, rt * 16)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(R8, RCX)
        asm.alu_ri("and", R8, 0xFFFFFFFC)   # aligned word address
        asm.mov_rr(R9, RCX)
        asm.alu_ri("and", R9, 3)
        asm.shift_ri("shl", R9, 3)          # s
        asm.mov_rr(RCX, R9)                 # shift count = s
        asm.mov_ri(R9, 1)
        asm.shift_cl("shl", R9)             # 1 << s
        asm.alu_ri("sub", R9, 1)            # low mask
        asm.ld32_ram(RAX, R14, R8)          # W
        asm.alu_rr("and", RAX, R9)          # W & mask
        asm.mov_rr(RCX, R9)
        asm.not_r(RCX)
        asm.alu_rr("and", RDX, RCX)         # rt & ~mask
        asm.alu_rr("or", RAX, RDX)
        asm.st32_ram(R14, R8, RAX)
    elif m == "swr":
        # LE SWR: store rt's low n bytes over W's low bytes at word_start..addr
        asm.ld32(RDX, R15, rt * 16)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(R8, RCX)
        asm.alu_ri("and", R8, 0xFFFFFFFC)   # aligned word address
        asm.mov_rr(R9, RCX)
        asm.alu_ri("and", R9, 3)
        asm.shift_ri("shl", R9, 3)          # s
        asm.mov_ri(RCX, 24)
        asm.alu_rr("sub", RCX, R9)          # 24 - s
        asm.mov_ri(R9, M32)
        asm.shift_cl("shr", R9)             # mask
        asm.ld32_ram(RAX, R14, R8)          # W
        asm.alu_rr("and", RDX, R9)          # rt & mask
        asm.mov_rr(RCX, R9)
        asm.not_r(RCX)
        asm.alu_rr("and", RAX, RCX)         # W & ~mask
        asm.alu_rr("or", RAX, RDX)
        asm.st32_ram(R14, R8, RAX)
    else:
        raise CodeGenError(f"unhandled instruction {m} at {insn.addr:#010x}")


def _fpr_ld(asm: Asm, xmm: int, index: int):
    asm.movd_ld(xmm, R15, FPR_OFF + index * 4)


def _fpr_st(asm: Asm, index: int, xmm: int):
    asm.movd_st(R15, FPR_OFF + index * 4, xmm)


def _fpu_binop(asm: Asm, method: str, dst_fpr: int, fs: int, ft: int,
               acc: bool = False):
    _fpr_ld(asm, 0, fs)
    _fpr_ld(asm, 1, ft)
    getattr(asm, method)(0, 1)
    if acc:
        asm.movd_st(R15, ACC_OFF, 0)
    else:
        _fpr_st(asm, dst_fpr, 0)


def emit_cop1(asm: Asm, insn: DecodedInsn):
    """Lower one COP1/FPU instruction (fields: fs=rd, ft=rt, fd=sa)."""
    m = insn.mnemonic
    fs, ft, fd = insn.rd, insn.rt, insn.shamt

    if m == "mtc1":
        _slot(asm, RAX, ft)
        asm.st32(R15, FPR_OFF + fs * 4, RAX)
    elif m == "mfc1":
        if ft:
            asm.ld32(RAX, R15, FPR_OFF + fs * 4)
            _sto(asm, ft, RAX)
    elif m == "cfc1":
        if ft:
            if fs == 31:
                asm.ld32(RAX, R15, FCR31_OFF)
            elif fs == 0:
                asm.mov_ri(RAX, 0x2E00)
            else:
                asm.mov_ri(RAX, 0)
            _sto(asm, ft, RAX)
    elif m == "ctc1":
        if fs == 31:
            _slot(asm, RAX, ft)
            asm.st32(R15, FCR31_OFF, RAX)
    elif m == "lwc1":
        _addr32(asm, insn.rs, insn.imm, RAX)
        asm.ld32_ram(RAX, R14, RCX)
        asm.st32(R15, FPR_OFF + ft * 4, RAX)
    elif m == "swc1":
        asm.ld32(RAX, R15, FPR_OFF + ft * 4)
        _addr32(asm, insn.rs, insn.imm, RDX)
        asm.st32_ram(R14, RCX, RAX)
    elif m in ("add.s", "sub.s", "mul.s", "div.s"):
        _fpu_binop(asm, {"add.s": "addss", "sub.s": "subss",
                         "mul.s": "mulss", "div.s": "divss"}[m], fd, fs, ft)
    elif m in ("adda.s", "suba.s", "mula.s"):
        _fpu_binop(asm, {"adda.s": "addss", "suba.s": "subss",
                         "mula.s": "mulss"}[m], 0, fs, ft, acc=True)
    elif m in ("madd.s", "msub.s"):
        asm.movd_ld(2, R15, ACC_OFF)
        _fpr_ld(asm, 0, fs)
        _fpr_ld(asm, 1, ft)
        asm.mulss(0, 1)
        asm.addss(2, 0) if m == "madd.s" else asm.subss(2, 0)
        _fpr_st(asm, fd, 2)
    elif m in ("madda.s", "msuba.s"):
        asm.movd_ld(2, R15, ACC_OFF)
        _fpr_ld(asm, 0, fs)
        _fpr_ld(asm, 1, ft)
        asm.mulss(0, 1)
        asm.addss(2, 0) if m == "madda.s" else asm.subss(2, 0)
        asm.movd_st(R15, ACC_OFF, 2)
    elif m == "mov.s":
        asm.ld32(RAX, R15, FPR_OFF + fs * 4)
        asm.st32(R15, FPR_OFF + fd * 4, RAX)
    elif m == "abs.s":
        asm.ld32(RAX, R15, FPR_OFF + fs * 4)
        asm.alu_ri("and", RAX, 0x7FFFFFFF)
        asm.st32(R15, FPR_OFF + fd * 4, RAX)
    elif m == "neg.s":
        asm.ld32(RAX, R15, FPR_OFF + fs * 4)
        asm.alu_ri("xor", RAX, 0x80000000)
        asm.st32(R15, FPR_OFF + fd * 4, RAX)
    elif m in ("sqrt.s", "rsqrt.s"):
        asm.ld32(RAX, R15, FPR_OFF + ft * 4)
        asm.alu_ri("and", RAX, 0x7FFFFFFF)
        asm.movd_xmm_gpr(0, RAX)
        asm.sqrtss(0, 0)
        if m == "rsqrt.s":
            _fpr_ld(asm, 1, fs)
            asm.divss(1, 0)
            _fpr_st(asm, fd, 1)
        else:
            _fpr_st(asm, fd, 0)
    elif m in ("max.s", "min.s"):
        _fpr_ld(asm, 0, fs)
        _fpr_ld(asm, 1, ft)
        asm.maxss(0, 1) if m == "max.s" else asm.minss(0, 1)
        _fpr_st(asm, fd, 0)
    elif m == "cvt.w.s":
        asm.ld32(RAX, R15, FPR_OFF + fs * 4)
        asm.mov_rr(RDX, RAX)
        asm.alu_ri("and", RDX, 0x7F800000)
        asm.alu_ri("cmp", RDX, 0x4E800000)
        clamp, positive, store = (asm.new_label() for _ in range(3))
        asm.jcc("a", clamp)
        asm.movd_xmm_gpr(0, RAX)
        asm.cvttss2si(RAX, 0)
        asm.jmp(store)
        asm.bind(clamp)
        asm.alu_ri("and", RAX, 0x80000000)
        asm.jcc("z", positive)
        asm.mov_ri(RAX, 0x80000000)
        asm.jmp(store)
        asm.bind(positive)
        asm.mov_ri(RAX, 0x7FFFFFFF)
        asm.bind(store)
        asm.st32(R15, FPR_OFF + fd * 4, RAX)
    elif m == "cvt.s.w":
        asm.ld32(RAX, R15, FPR_OFF + fs * 4)
        asm.cvtsi2ss(0, RAX)
        asm.movd_st(R15, FPR_OFF + fd * 4, 0)
    elif m == "c.f":
        asm.ld32(RAX, R15, FCR31_OFF)
        asm.alu_ri("and", RAX, 0xFF7FFFFF)
        asm.st32(R15, FCR31_OFF, RAX)
    elif m in ("c.eq", "c.lt", "c.le"):
        _fpr_ld(asm, 0, fs)
        _fpr_ld(asm, 1, ft)
        asm.ucomiss(0, 1)
        asm.setcc({"c.eq": "e", "c.lt": "b", "c.le": "be"}[m], RDX)
        asm.setcc("np", RAX)
        asm.movzx(R8, RDX)
        asm.movzx(R9, RAX)
        asm.alu_rr("and", R8, R9)
        asm.shift_ri("shl", R8, 23)
        asm.ld32(RAX, R15, FCR31_OFF)
        asm.alu_ri("and", RAX, 0xFF7FFFFF)
        asm.alu_rr("or", RAX, R8)
        asm.st32(R15, FCR31_OFF, RAX)
    else:
        raise CodeGenError(
            f"COP1 instruction {m} not lowered to x64 yet at {insn.addr:#010x}")


_MMI_SSE = {
    # wrapping add/sub
    "paddb": "paddb", "paddh": "paddw", "paddw": "paddd",
    "psubb": "psubb", "psubh": "psubw", "psubw": "psubd",
    # signed/unsigned saturation
    "paddsb": "paddsb", "paddsh": "paddsw",
    "psubsb": "psubsb", "psubsh": "psubsw",
    "paddub": "paddusb", "padduh": "paddusw",
    "psubub": "psubusb", "psubuh": "psubusw",
    # bitwise
    "pand": "pand", "por": "por", "pxor": "pxor",
    # compares / min / max
    "pceqb": "pcmpeqb", "pceqh": "pcmpeqw", "pceqw": "pcmpeqd",
    "pcgtb": "pcmpgtb", "pcgth": "pcmpgtw", "pcgtw": "pcmpgtd",
    "pminh": "pminsw", "pmaxh": "pmaxsw",
}
_MMI_INTERLEAVE = {
    "pextlw": ("punpckldq", 1, 0, 1),
    "pextuw": ("punpckhdq", 1, 0, 1),
    "pextlh": ("punpcklwd", 1, 0, 1),
    "pextuh": ("punpckhwd", 1, 0, 1),
    "pextlb": ("punpcklbw", 1, 0, 1),
    "pextub": ("punpckhbw", 1, 0, 1),
    "pcpyld": ("punpcklqdq", 1, 0, 1),
    "pcpyud": ("punpckhqdq", 0, 1, 0),
}
_MMI_SHUFFLE = {
    "pcpyh": ((0, 0), (0, 0)),
    "pexeh": ((0xC6, 0xC6), None),
    "prevh": ((0x1B, 0x1B), None),
    "pexch": ((0xD8, 0xD8), None),
    "pexew": (0xC6, None),
    "prot3w": (0xC9, None),
    "pexcw": (0xD8, None),
}


def emit_mmi(asm: Asm, insn: DecodedInsn):
    """Lower one MMI instruction with SSE2 (128-bit GPR slots)."""
    m = insn.mnemonic
    rs, rt, rd = insn.rs, insn.rt, insn.rd

    if m in _MMI_SSE:
        _xmm_slot(asm, 0, rs)
        _xmm_slot(asm, 1, rt)
        getattr(asm, _MMI_SSE[m])(0, 1)
        _xmm_store(asm, rd, 0)
    elif m == "pnor":
        _xmm_slot(asm, 0, rs)
        _xmm_slot(asm, 1, rt)
        asm.por(0, 1)
        asm.pcmpeqd(2, 2)
        asm.pxor(0, 2)
        _xmm_store(asm, rd, 0)
    elif m in _MMI_INTERLEAVE:
        method, dst, src, out = _MMI_INTERLEAVE[m]
        _xmm_slot(asm, 0, rs)
        _xmm_slot(asm, 1, rt)
        getattr(asm, method)(dst, src)
        _xmm_store(asm, rd, out)
    elif m in _MMI_SHUFFLE:
        _xmm_slot(asm, 1, rt)
        first, second = _MMI_SHUFFLE[m]
        if isinstance(first, tuple):
            asm.pshuflw(1, 1, first[0])
            asm.pshufhw(1, 1, first[1])
        else:
            asm.pshufd(1, 1, first)
        _xmm_store(asm, rd, 1)
    elif m == "plzcw":
        # count leading sign bits per low 32-bit lane, minus one
        def lane(load):
            load()
            nonneg, zero, done = (asm.new_label() for _ in range(3))
            asm.test_rr(RAX, RAX)
            asm.jcc("ns", nonneg)
            asm.not_r(RAX)
            asm.bind(nonneg)
            asm.test_rr(RAX, RAX)
            asm.jcc("z", zero)
            asm.bsr(RDX, RAX)
            asm.mov_ri(RAX, 30)
            asm.alu_rr("sub", RAX, RDX)
            asm.jmp(done)
            asm.bind(zero)
            asm.mov_ri(RAX, 31)
            asm.bind(done)

        lane(lambda: _slot(asm, RAX, rs))
        asm.mov_rr(R8, RAX)
        lane(lambda: (_slot64(asm, RAX, rs), asm.shr64_ri(RAX, 32)))
        asm.mov_rr(RDX, RAX)
        asm.shl64_ri(RDX, 32)
        asm.mov_rr(RAX, R8)
        asm.alu_rr64("or", RAX, RDX)
        if rd:
            asm.st64(R15, rd * 16, RAX)
            asm.mov_ri(RDX, 0)
            asm.st64(R15, rd * 16 + 8, RDX)
    elif m == "mult1":
        _slot(asm, RAX, rs)
        asm.movsxd(RAX, RAX)
        _slot(asm, RCX, rt)
        asm.movsxd(RCX, RCX)
        asm.imul64_rr(RAX, RCX)
        asm.mov_rr(RDX, RAX)
        asm.movsxd(RDX, RDX)
        asm.st64(R15, LO_OFF + 8, RDX)
        asm.mov_rr(RDX, RAX)
        asm.shift_ri64("sar", RDX, 32)
        asm.movsxd(RDX, RDX)
        asm.st64(R15, HI_OFF + 8, RDX)
        _sto(asm, rd, RAX)
    elif m in ("mfhi1", "mflo1"):
        asm.ld64_rm(RAX, R15, (HI_OFF if m == "mfhi1" else LO_OFF) + 8)
        _sto64(asm, rd, RAX)
    elif m in ("mthi1", "mtlo1"):
        _slot64(asm, RAX, rs)
        asm.st64(R15, (HI_OFF if m == "mthi1" else LO_OFF) + 8, RAX)
    else:
        raise CodeGenError(
            f"MMI instruction {m} not lowered to x64 yet at {insn.addr:#010x}")


def emit_terminator(asm: Asm, block, labels: dict[int, Label],
                    epilogue: Label, entry_labels: dict[int, Label]):
    term = block.term
    if isinstance(term, BranchTerm):
        insn = term.insn
        m = insn.mnemonic
        if m.startswith("bc1"):
            asm.ld32(RAX, R15, FCR31_OFF)
            asm.alu_ri("and", RAX, 0x00800000)
            asm.setcc("e" if m in ("bc1f", "bc1fl") else "ne", R10)
            asm.movzx(R10, R10)
            if term.insn.is_likely:
                skip = asm.new_label()
                asm.test_rr(R10, R10)
                asm.jcc("e", skip)
                emit_insn(asm, term.delay)
                asm.jmp(_label(labels, term.target, asm))
                asm.bind(skip)
                asm.jmp(_label(labels, term.fallthrough, asm))
            else:
                emit_insn(asm, term.delay)
                asm.test_rr(R10, R10)
                asm.jcc("ne", _label(labels, term.target, asm))
                asm.jmp(_label(labels, term.fallthrough, asm))
            return
        if m in ("beq", "bne", "beql", "bnel"):
            _slot64(asm, RAX, insn.rs)
            _slot64(asm, RCX, insn.rt)
            asm.alu_rr64("cmp", RAX, RCX)
            cc = "e" if m in ("beq", "beql") else "ne"
        else:
            _slot64(asm, RAX, insn.rs)
            asm.alu_ri64("cmp", RAX, 0)
            cc = _BRANCH_CC[m]
        asm.setcc(cc, R10)
        asm.movzx(R10, R10)
        lik = term.insn.is_likely
        link = term.insn.is_branch_link
        if lik or link:
            # A branch-likely nullifies its delay slot when not taken, and a
            # REGIMM branch-and-link writes $ra only when taken: both need
            # the condition tested before the delay slot is emitted.
            skip = asm.new_label()
            if lik:
                asm.test_rr(R10, R10)
                asm.jcc("e", skip)
            if link:
                if not lik:
                    no_link = asm.new_label()
                    asm.test_rr(R10, R10)
                    asm.jcc("e", no_link)
                asm.mov_ri(RAX, (term.insn.addr + 8) & M32)
                _sto(asm, 31, RAX)
                if not lik:
                    asm.bind(no_link)
            emit_insn(asm, term.delay)
            if lik:
                asm.jmp(_label(labels, term.target, asm))
                asm.bind(skip)
                asm.jmp(_label(labels, term.fallthrough, asm))
            else:
                asm.test_rr(R10, R10)  # the delay slot may clobber flags
                asm.jcc("ne", _label(labels, term.target, asm))
                asm.jmp(_label(labels, term.fallthrough, asm))
        else:
            emit_insn(asm, term.delay)
            asm.test_rr(R10, R10)
            asm.jcc("ne", _label(labels, term.target, asm))
            asm.jmp(_label(labels, term.fallthrough, asm))
    elif isinstance(term, JumpTerm):
        emit_insn(asm, term.delay)
        asm.jmp(_label(labels, term.target, asm))
    elif isinstance(term, CallTerm):
        asm.mov_ri(RAX, (term.insn.addr + 8) & M32)  # link before delay slot
        _sto(asm, 31, RAX)
        emit_insn(asm, term.delay)
        if term.target not in entry_labels:
            raise CodeGenError(
                f"call target {term.target:#010x} is not a compiled function")
        asm.mov_rr(RCX, R15)  # rcx is scratch; callee prologue reloads r15 from it
        asm.call(entry_labels[term.target])  # enter at the callee PROLOGUE
        asm.jmp(_label(labels, term.fallthrough, asm))
    elif isinstance(term, ReturnTerm):
        emit_insn(asm, term.delay)
        asm.mov_ri(RAX, EXIT_NORMAL)
        asm.jmp(epilogue)
    elif isinstance(term, IndirectTerm):
        raise CodeGenError(
            f"indirect jump (jr ${term.insn.rs}) unsupported at {term.insn.addr:#010x}"
        )
    elif isinstance(term, StopTerm):
        asm.mov_ri(RAX, EXIT_NORMAL)
        asm.jmp(epilogue)
    elif isinstance(term, FallTo):
        asm.jmp(_label(labels, term.target, asm))
    else:
        raise CodeGenError(f"missing terminator in block {block.addr:#010x}")


def _label(labels: dict[int, Label], addr: int, asm: Asm) -> Label:
    lbl = labels.get(addr)
    if lbl is None:
        raise CodeGenError(f"branch target {addr:#010x} is not a compiled block")
    return lbl


def compile_program(insn_at, entry: int, stop_addrs: set[int]) -> Program:
    """Compile a recompiled region (main + callees) to native x64."""
    # Discover all callee entries (jal targets) to a fixpoint first, then
    # rebuild every function with the complete start set. Without this, a
    # callee entry discovered mid-scan would be inlined into its caller AND
    # built separately (duplicate labels -> corrupt jumps).
    function_starts: set[int] = {entry}
    while True:
        grew = False
        for e in sorted(function_starts):
            fn = build_function(insn_at, e, lambda addr: addr in function_starts,
                                stop_addrs if e == entry else set())
            for blk in fn.blocks.values():
                t = blk.term
                if isinstance(t, CallTerm) and t.target not in function_starts:
                    function_starts.add(t.target)
                    grew = True
        if not grew:
            break
    functions: dict[int, Function] = {}
    for e in sorted(function_starts):
        functions[e] = build_function(insn_at, e,
                                      lambda addr: addr in function_starts,
                                      stop_addrs if e == entry else set())

    asm = Asm()
    labels = {addr: asm.new_label()
              for fn in functions.values() for addr in fn.blocks}
    entry_labels = {faddr: asm.new_label() for faddr in functions}
    offsets: dict[int, int] = {}
    for faddr, fn in functions.items():
        is_main = faddr == entry
        offsets[faddr] = len(asm)
        asm.bind(entry_labels[faddr])
        asm.push(R14)
        asm.push(R15)
        asm.mov_rr(R15, RCX)
        asm.ld64_rm(R14, R15, RAM_OFF)
        epilogue = asm.new_label()
        sys_exit = epilogue if is_main else None
        for baddr in sorted(fn.blocks):
            asm.bind(labels[baddr])
            blk = fn.blocks[baddr]
            for _, insn in blk.insns:
                emit_insn(asm, insn, sys_exit)
            emit_terminator(asm, blk, labels, epilogue, entry_labels)
        asm.bind(epilogue)
        asm.pop(R15)
        asm.pop(R14)
        asm.ret()
    return Program(code=asm.finalize(), entry_offset=offsets[entry],
                   function_offsets=offsets)
