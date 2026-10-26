"""R5900 COP1 (FPU) semantics for the golden interpreter (T3.6).

The implementation follows PCSX2's FPU.cpp, including the model quirks the
PS2 relies on:

- denormal operands are treated as signed zero, infinities as +/-FLT_MAX
  (`fpuDouble`)
- arithmetic results that overflow are clamped to +/-FLT_MAX, denormal
  results to signed zero, and the FCR31 cause flags are updated
- division by zero (or a denormal divisor) clamps to +/-FLT_MAX rather
  than producing IEEE infinity
- CVT.W.S truncates and saturates outside the representable range
- C.cond sets FCR31 bit 23, which BC1F/BC1T then test

COP1 field layout: fs = rd, ft = rt, fd = sa (the normal MIPS roles do
not apply). MFC1/CFC1/MTC1/CTC1 use rt as the GPR and rd as the FPU
register.
"""
from __future__ import annotations

import math
import struct

from ..r5900.decoder import DecodedInsn
from .interpreter import M32, sext32

POS_FMAX = 0x7F7FFFFF
NEG_FMAX = 0xFF7FFFFF

FPUflagC = 0x00800000
FPUflagI = 0x00020000
FPUflagD = 0x00010000
FPUflagO = 0x00008000
FPUflagU = 0x00004000
FPUflagSI = 0x00000040
FPUflagSD = 0x00000020
FPUflagSO = 0x00000010
FPUflagSU = 0x00000008


def f32(raw: int) -> float:
    return struct.unpack("<f", (raw & M32).to_bytes(4, "little"))[0]


def bits(value: float) -> int:
    try:
        return int.from_bytes(struct.pack("<f", value), "little")
    except OverflowError:
        return POS_FMAX if value > 0 else NEG_FMAX


def fpu_double(raw: int) -> float:
    """PCSX2's operand view: flush denormals, clamp infinities."""
    exponent = raw & 0x7F800000
    if exponent == 0:
        return f32(raw & 0x80000000)
    if exponent == 0x7F800000:
        return f32((raw & 0x80000000) | POS_FMAX)
    return f32(raw)


def _check_overflow(st, raw: int) -> int:
    if (raw & 0x7FFFFFFF) == 0x7F800000:
        st.fcr31 |= FPUflagO | FPUflagSO
        return (raw & 0x80000000) | POS_FMAX
    st.fcr31 &= ~FPUflagO
    return raw


def _check_underflow(st, raw: int) -> int:
    if (raw & 0x7F800000) == 0 and (raw & 0x007FFFFF) != 0:
        st.fcr31 |= FPUflagU | FPUflagSU
        return raw & 0x80000000
    st.fcr31 &= ~FPUflagU
    return raw


def _arith(st, a: int, b: int, op: str) -> int:
    if op == "div" and (b & 0x7F800000) == 0:
        # PCSX2 clamps instead of returning IEEE infinity
        st.fcr31 |= (FPUflagI | FPUflagSI) if (a & 0x7F800000) == 0 \
            else (FPUflagD | FPUflagSD)
        return ((a ^ b) & 0x80000000) | POS_FMAX
    x, y = fpu_double(a), fpu_double(b)
    if op == "add":
        raw = bits(x + y)
    elif op == "sub":
        raw = bits(x - y)
    elif op == "mul":
        raw = bits(x * y)
    else:
        raw = bits(x / y)
    return _check_underflow(st, _check_overflow(st, raw))


def _compare(st, a: int, b: int, op: str) -> None:
    x, y = fpu_double(a), fpu_double(b)
    if math.isnan(x) or math.isnan(y):
        result = False
        st.fcr31 |= FPUflagI | FPUflagSI
    else:
        result = {"eq": x == y, "lt": x < y, "le": x <= y}[op]
    st.fcr31 = (st.fcr31 | FPUflagC) if result else (st.fcr31 & ~FPUflagC)


def fp_max(a: int, b: int) -> int:
    sa = a - 0x100000000 if a & 0x80000000 else a
    sb = b - 0x100000000 if b & 0x80000000 else b
    return min(sa, sb) & M32 if sa < 0 and sb < 0 else max(sa, sb) & M32


def fp_min(a: int, b: int) -> int:
    sa = a - 0x100000000 if a & 0x80000000 else a
    sb = b - 0x100000000 if b & 0x80000000 else b
    return max(sa, sb) & M32 if sa < 0 and sb < 0 else min(sa, sb) & M32


def execute_cop1(st, insn: DecodedInsn, mem) -> None:
    """Execute one decoded COP1 / FPU-memory instruction."""
    m = insn.mnemonic
    fs, ft, fd = insn.rd, insn.rt, insn.shamt
    rt = insn.rt

    if m == "mfc1":
        st._set(rt, sext32(st.fpr[fs]))
    elif m == "cfc1":
        if fs == 31:
            st._set(rt, sext32(st.fcr31))
        elif fs == 0:
            st._set(rt, 0x2E00)
        else:
            st._set(rt, 0)
    elif m == "mtc1":
        st.fpr[fs] = st._get(rt) & M32
    elif m == "ctc1":
        if fs == 31:
            st.fcr31 = st._get(rt) & M32
    elif m == "lwc1":
        addr = (st._get(insn.rs) + insn.imm) & M32
        st.fpr[rt] = mem.read32(addr)
    elif m == "swc1":
        addr = (st._get(insn.rs) + insn.imm) & M32
        mem.write32(addr, st.fpr[rt])
    elif m in ("add.s", "sub.s", "mul.s", "div.s"):
        st.fpr[fd] = _arith(st, st.fpr[fs], st.fpr[ft], m[:3])
    elif m in ("adda.s", "suba.s", "mula.s"):
        st.fpu_acc = _arith(st, st.fpr[fs], st.fpr[ft], m[:3])
    elif m in ("madd.s", "msub.s"):
        temp = _arith(st, st.fpr[fs], st.fpr[ft], "mul")
        st.fpr[fd] = _arith(st, st.fpu_acc, temp, "add" if m == "madd.s" else "sub")
    elif m in ("madda.s", "msuba.s"):
        temp = _arith(st, st.fpr[fs], st.fpr[ft], "mul")
        st.fpu_acc = _arith(st, st.fpu_acc, temp, "add" if m == "madda.s" else "sub")
    elif m == "mov.s":
        st.fpr[fd] = st.fpr[fs]
    elif m == "abs.s":
        st.fpr[fd] = st.fpr[fs] & 0x7FFFFFFF
    elif m == "neg.s":
        st.fpr[fd] = st.fpr[fs] ^ 0x80000000
    elif m == "sqrt.s":
        raw = st.fpr[ft]
        if (raw & 0x7F800000) == 0:
            st.fpr[fd] = raw & 0x80000000
        elif raw & 0x80000000:
            st.fcr31 |= FPUflagI | FPUflagSI
            st.fpr[fd] = bits(math.sqrt(abs(fpu_double(raw))))
        else:
            st.fpr[fd] = bits(math.sqrt(fpu_double(raw)))
    elif m == "rsqrt.s":
        raw = st.fpr[ft]
        if (raw & 0x7F800000) == 0:
            st.fcr31 |= FPUflagD | FPUflagSD
            st.fpr[fd] = (raw & 0x80000000) | POS_FMAX
        elif raw & 0x80000000:
            st.fcr31 |= FPUflagI | FPUflagSI
            st.fpr[fd] = _arith(st, st.fpr[fs],
                                bits(math.sqrt(abs(fpu_double(raw)))), "div")
        else:
            st.fpr[fd] = _arith(st, st.fpr[fs],
                                bits(math.sqrt(fpu_double(raw))), "div")
    elif m == "max.s":
        st.fpr[fd] = fp_max(st.fpr[fs], st.fpr[ft])
        st.fcr31 &= ~(FPUflagO | FPUflagU)
    elif m == "min.s":
        st.fpr[fd] = fp_min(st.fpr[fs], st.fpr[ft])
        st.fcr31 &= ~(FPUflagO | FPUflagU)
    elif m == "cvt.w.s":
        raw = st.fpr[fs]
        if (raw & 0x7F800000) <= 0x4E800000:
            value = fpu_double(raw)
            st.fpr[fd] = int(value) & M32
        else:
            st.fpr[fd] = 0x7FFFFFFF if not (raw & 0x80000000) else 0x80000000
    elif m == "cvt.s.w":
        value = st.fpr[fs]
        if value & 0x80000000:
            value -= 0x100000000
        st.fpr[fd] = bits(float(value))
    elif m == "c.f":
        st.fcr31 &= ~FPUflagC
    elif m in ("c.eq", "c.lt", "c.le"):
        _compare(st, st.fpr[fs], st.fpr[ft], m[2:])
    else:
        from .interpreter import UnsupportedInstruction
        raise UnsupportedInstruction(insn.addr, insn.raw, m)
