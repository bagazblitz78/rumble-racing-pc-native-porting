"""x64 code generation from recompiled EE CFGs (task T2.7 core).

Generated-code calling convention (self-contained, closed world):
  - one native function `int run(void* ctx)` (Win64: ctx in rcx)
  - ctx layout (offsets): regs[i] = i*8 (u64 slots, low 32 used), hi = 256,
    lo = 264, guest_ram_ptr = 272
  - r15 = ctx pointer, r14 = guest RAM base, r10 = branch-condition scratch;
    rax/rcx/rdx/r8/r9 are general scratch
  - returns -1 (0xFFFFFFFF) on normal completion, else the syscall code
  - generated code never calls OS APIs directly; it never uses aligned
    stack slots or SSE, so no ABI shadow-space dance is needed between
    generated functions (documented deviation from Win64 ABI, safe because
    only generated code is ever called).

Semantics mirror tools/ps2toolkit/golden/interpreter.py exactly,
including PS2 DIV quirks and big-endian LWL/LWR/SWL/SWR pair rules.
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
HI_OFF = 256
LO_OFF = 264
RAM_OFF = 272
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
    asm.ld32(dst, R15, reg * 8)


def _slot64(asm: Asm, dst: int, reg: int):
    """ld64 dst <- GPR[reg]."""
    asm.ld64_rm(dst, R15, reg * 8)


def _sto(asm: Asm, reg: int, src: int):
    """GPR[reg] <- sign-extended 32-bit src ($zero writes discarded)."""
    if reg == 0:
        return
    asm.movsxd(src, src)
    asm.st64(R15, reg * 8, src)


def _sto64(asm: Asm, reg: int, src: int):
    """GPR[reg] <- full 64-bit src ($zero writes discarded)."""
    if reg == 0:
        return
    asm.st64(R15, reg * 8, src)


def _addr32(asm: Asm, base_reg: int, imm: int, dst: int):
    """dst <- (GPR[base_reg] + imm) in RCX, masked to physical."""
    asm.ld32(dst, R15, base_reg * 8)
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
    elif m in ("andi", "ori", "xori"):
        op = {"andi": "and", "ori": "or", "xori": "xor"}[m]
        _slot64(asm, RAX, rs)
        asm.alu_ri64(op, RAX, insn.uimm)
        _sto64(asm, rt, RAX)
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
        _slot64(asm, RAX, rs)
        _slot64(asm, RCX, rt)
        asm.alu_rr64(m, RAX, RCX)
        _sto64(asm, rd, RAX)
    elif m == "nor":
        _slot64(asm, RAX, rs)
        _slot64(asm, RCX, rt)
        asm.alu_rr64("or", RAX, RCX)
        asm.not_r64(RAX)
        _sto64(asm, rd, RAX)
    elif m in ("movz", "movn"):
        _slot64(asm, RAX, rs)
        _slot64(asm, RCX, rt)
        skip = asm.new_label()
        asm.test_rr64(RCX, RCX)
        # movz stores when rt == 0; movn when rt != 0
        asm.jcc("ne" if m == "movz" else "e", skip)
        _sto64(asm, rd, RAX)
        asm.bind(skip)
    elif m in ("sync", "cache", "pref"):
        return  # no cache/coherence model in the single-threaded recompiler
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
        asm.movzx16(RAX, RDX)
        asm.xchg_al_ah()
        asm.movsx16(RAX, RAX)
        _sto(asm, rt, RAX)
    elif m == "lhu":
        _slot(asm, RAX, rs)
        asm.alu_ri("add", RAX, insn.imm)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 0x1FFFFFFF)
        asm.ld16_ram(RDX, R14, RCX)
        asm.movzx16(RAX, RDX)
        asm.xchg_al_ah()
        _sto(asm, rt, RAX)
    elif m == "lw":
        _addr32(asm, rs, insn.imm, RAX)
        asm.ld32_ram(RAX, R14, RCX)
        asm.bswap(RAX)
        _sto(asm, rt, RAX)
    elif m == "lwu":
        _addr32(asm, rs, insn.imm, RAX)
        asm.ld32_ram(RAX, R14, RCX)
        asm.bswap(RAX)
        _sto64(asm, rt, RAX)
    elif m == "ld":
        _addr32(asm, rs, insn.imm, RAX)
        asm.ld64_ram(RAX, R14, RCX)
        asm.bswap64(RAX)
        _sto64(asm, rt, RAX)
    elif m == "sb":
        _slot(asm, RDX, rt)
        _addr32(asm, rs, insn.imm, RAX)
        asm.st8_ram(R14, RCX, RDX)
    elif m == "sh":
        _slot(asm, RAX, rt)
        asm.xchg_al_ah()
        _addr32(asm, rs, insn.imm, RDX)
        asm.st16_ram(R14, RCX, RAX)
    elif m == "sw":
        _slot(asm, RDX, rt)
        asm.bswap(RDX)  # guest memory is big-endian
        _addr32(asm, rs, insn.imm, RAX)
        asm.st32_ram(R14, RCX, RDX)
    elif m == "sd":
        _slot64(asm, RDX, rt)
        asm.bswap64(RDX)
        _addr32(asm, rs, insn.imm, RAX)
        asm.st64_ram(R14, RCX, RDX)
    elif m == "lwl":
        # rt <- ((W & (M32 >> s)) << s) | (rt & ~(M32 >> s)), s = 8*(addr&3)
        asm.ld32(R8, R15, rt * 8)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(RDX, RCX)
        asm.alu_ri("and", RDX, 3)
        asm.shift_ri("shl", RDX, 3)
        asm.mov_rr(R9, RDX)  # s
        asm.alu_ri("and", RCX, 0xFFFFFFFC)
        asm.ld32_ram(RAX, R14, RCX)
        asm.bswap(RAX)
        asm.mov_rr(RCX, R9)  # cl = s
        asm.mov_ri(RDX, 0xFFFFFFFF)
        asm.shift_cl("shr", RDX)  # lowmask = M32 >> s
        asm.alu_rr("and", RAX, RDX)
        asm.shift_cl("shl", RAX)  # (W & lowmask) << s
        asm.mov_ri(RDX, 0xFFFFFFFF)
        asm.shift_cl("shl", RDX)
        asm.not_r(RDX)  # keep = low s bits = ~(M32 << s)
        asm.alu_rr("and", R8, RDX)
        asm.alu_rr("or", RAX, R8)
        _sto(asm, rt, RAX)
    elif m == "lwr":
        # rt <- (rt - (rt & lowmask)) | (W >> t), t = 24 - 8*(addr&3)
        asm.ld32(R8, R15, rt * 8)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 3)
        asm.shift_ri("shl", RCX, 3)
        asm.mov_ri(R9, 24)
        asm.alu_rr("sub", R9, RCX)
        asm.alu_ri("and", RAX, 0xFFFFFFFC)
        asm.ld32_ram(RAX, R14, RAX)
        asm.bswap(RAX)
        asm.mov_rr(RCX, R9)
        asm.shift_cl("shr", RAX)
        asm.mov_ri(RDX, 0xFFFFFFFF)
        asm.shift_cl("shr", RDX)
        asm.alu_rr("and", R8, RDX)
        asm.ld32(RDX, R15, rt * 8)
        asm.alu_rr("sub", RDX, R8)
        asm.alu_rr("or", RAX, RDX)
        _sto(asm, rt, RAX)
    elif m == "swl":
        # W' <- (W & ~(M32 >> s)) | (rt >> s)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 3)
        asm.shift_ri("shl", RCX, 3)
        asm.alu_ri("and", RAX, 0xFFFFFFFC)
        asm.mov_rr(R9, RAX)
        asm.ld32_ram(RAX, R14, RAX)
        asm.bswap(RAX)
        asm.mov_ri(R8, 0xFFFFFFFF)
        asm.shift_cl("shr", R8)
        asm.not_r(R8)
        asm.alu_rr("and", RAX, R8)
        _slot(asm, RDX, rt)
        asm.shift_cl("shr", RDX)
        asm.alu_rr("or", RAX, RDX)
        asm.bswap(RAX)
        asm.st32_ram(R14, R9, RAX)
    elif m == "swr":
        # W' <- (W & ~(M32 << t)) | ((rt & (M32 >> t)) << t), t = 24 - 8*(addr&3)
        _addr32(asm, rs, insn.imm, RAX)
        asm.mov_rr(RCX, RAX)
        asm.alu_ri("and", RCX, 3)
        asm.shift_ri("shl", RCX, 3)
        asm.alu_ri("and", RAX, 0xFFFFFFFC)
        asm.mov_rr(R8, RAX)
        asm.ld32_ram(RAX, R14, RAX)
        asm.bswap(RAX)
        asm.mov_ri(R9, 24)
        asm.alu_rr("sub", R9, RCX)
        asm.mov_rr(RCX, R9)
        asm.mov_ri(R9, 0xFFFFFFFF)
        asm.shift_cl("shr", R9)
        _slot(asm, RDX, rt)
        asm.alu_rr("and", RDX, R9)
        asm.shift_cl("shl", RDX)
        asm.mov_ri(R9, 0xFFFFFFFF)
        asm.shift_cl("shl", R9)
        asm.not_r(R9)
        asm.alu_rr("and", RAX, R9)
        asm.alu_rr("or", RAX, RDX)
        asm.bswap(RAX)
        asm.st32_ram(R14, R8, RAX)
    else:
        raise CodeGenError(f"unhandled instruction {m} at {insn.addr:#010x}")


def emit_terminator(asm: Asm, block, labels: dict[int, Label],
                    epilogue: Label, entry_labels: dict[int, Label]):
    term = block.term
    if isinstance(term, BranchTerm):
        insn = term.insn
        m = insn.mnemonic
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
