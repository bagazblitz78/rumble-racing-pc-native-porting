"""R5900 MMI (multimedia) semantics for the golden interpreter (T3.5).

Encoding (from PCSX2's R5900 opcode tables):

    opcode 0x1C
    funct 0x08 -> MMI0,  funct 0x28 -> MMI1,
    funct 0x09 -> MMI2,  funct 0x29 -> MMI3
    the low opcode within a group is the `sa` field (bits 10:6)

Registers are 128-bit; lane 0 is the least-significant lane, matching the
usual `GPR.UL[0]` naming. Operations are implemented as pure functions
over Python ints, so they are easy to cross-check against PCSX2's
interpreter code.

Only the instructions Rumble Racing actually reaches (plus their close
relatives) are implemented; anything else raises UnsupportedInstruction so
the boot tracer reports it precisely.
"""
from __future__ import annotations

from ..r5900.decoder import DecodedInsn
from .interpreter import M32, M64, M128, UnsupportedInstruction, s64, sext32, sext64

MASK = {8: 0xFF, 16: 0xFFFF, 32: M32, 64: M64, 128: M128}


def lanes(value: int, bits: int) -> list[int]:
    mask = MASK[bits]
    return [(value >> (bits * i)) & mask for i in range(128 // bits)]


def pack(values: list[int], bits: int) -> int:
    mask = MASK[bits]
    out = 0
    for i, v in enumerate(values):
        out |= (v & mask) << (bits * i)
    return out


def slane(value: int, bits: int) -> int:
    mask = MASK[bits]
    value &= mask
    return value - (mask + 1) if value & (mask ^ (mask >> 1)) else value


def _sat_signed(value: int, bits: int) -> int:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return lo if value < lo else hi if value > hi else value


def _sat_unsigned(value: int, bits: int) -> int:
    return 0 if value < 0 else MASK[bits] if value > MASK[bits] else value


def _binary_lanes(rs: int, rt: int, bits: int, fn) -> int:
    return pack([fn(a, b) for a, b in zip(lanes(rs, bits), lanes(rt, bits))], bits)


def _unary_lanes(rt: int, bits: int, fn) -> int:
    return pack([fn(a) for a in lanes(rt, bits)], bits)


def _compare(rs: int, rt: int, bits: int, fn) -> int:
    mask = MASK[bits]
    return pack([mask if fn(a, b) else 0
                 for a, b in zip(lanes(rs, bits), lanes(rt, bits))], bits)


def _interleave(rs: int, rt: int, bits: int, start: int, count: int) -> int:
    """Interleave `count` lanes starting at `start`: rt, rs, rt, rs, ..."""
    rl, tl = lanes(rs, bits), lanes(rt, bits)
    out = []
    for i in range(start, start + count):
        out.append(tl[i])
        out.append(rl[i])
    return pack(out, bits)


def _abs_lane(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    if value == sign:
        return sign - 1
    return -value if value & sign else value


def _count_leading_sign_bits(value: int) -> int:
    """PCSX2 Common::CountLeadingSignBits (including the sign bit)."""
    v = value & M32
    if v & 0x80000000:
        v = (~v) & M32
    return 32 if v == 0 else 32 - v.bit_length()


def _shift_lanes(value: int, bits: int, amount: int, kind: str) -> int:
    def one(lane: int) -> int:
        if kind == "sll":
            return lane << amount
        if kind == "srl":
            return lane >> amount
        return slane(lane, bits) >> amount
    return _unary_lanes(value, bits, one)


def _set_hi_lo(st, which: str, slot: int, value: int):
    """Write one 64-bit half (slot 0 = low, 1 = high) of 128-bit HI/LO."""
    shift = 64 * slot
    old = getattr(st, which)
    setattr(st, which, (old & ~(M64 << shift)) | ((value & M64) << shift))


def execute_mmi(st, insn: DecodedInsn) -> None:
    """Execute one decoded MMI instruction against `st`."""
    m = insn.mnemonic.upper()
    rs, rt, rd = insn.rs, insn.rt, insn.rd
    if rd == 0 and m not in ("PMTHI", "PMTLO", "MTHI1", "MTLO1",
                             "PMTHL", "MADD", "MADDU", "MADD1", "MADDU1"):
        return  # writes to $zero are discarded (reads/writes of HI/LO aside)

    a, b = st._get(rs), st._get(rt)

    # --- packed add/sub (wrapping) ---
    if m == "PADDW":
        out = _binary_lanes(a, b, 32, lambda x, y: x + y)
    elif m == "PSUBW":
        out = _binary_lanes(a, b, 32, lambda x, y: x - y)
    elif m == "PADDH":
        out = _binary_lanes(a, b, 16, lambda x, y: x + y)
    elif m == "PSUBH":
        out = _binary_lanes(a, b, 16, lambda x, y: x - y)
    elif m == "PADDB":
        out = _binary_lanes(a, b, 8, lambda x, y: x + y)
    elif m == "PSUBB":
        out = _binary_lanes(a, b, 8, lambda x, y: x - y)
    # --- saturating add/sub ---
    elif m == "PADDSW":
        out = _binary_lanes(a, b, 32, lambda x, y: _sat_signed(slane(x, 32) + slane(y, 32), 32))
    elif m == "PSUBSW":
        out = _binary_lanes(a, b, 32, lambda x, y: _sat_signed(slane(x, 32) - slane(y, 32), 32))
    elif m == "PADDSH":
        out = _binary_lanes(a, b, 16, lambda x, y: _sat_signed(slane(x, 16) + slane(y, 16), 16))
    elif m == "PSUBSH":
        out = _binary_lanes(a, b, 16, lambda x, y: _sat_signed(slane(x, 16) - slane(y, 16), 16))
    elif m == "PADDSB":
        out = _binary_lanes(a, b, 8, lambda x, y: _sat_signed(slane(x, 8) + slane(y, 8), 8))
    elif m == "PSUBSB":
        out = _binary_lanes(a, b, 8, lambda x, y: _sat_signed(slane(x, 8) - slane(y, 8), 8))
    elif m == "PADDUW":
        out = _binary_lanes(a, b, 32, lambda x, y: _sat_unsigned(x + y, 32))
    elif m == "PSUBUW":
        out = _binary_lanes(a, b, 32, lambda x, y: _sat_unsigned(x - y, 32))
    elif m == "PADDUH":
        out = _binary_lanes(a, b, 16, lambda x, y: _sat_unsigned(x + y, 16))
    elif m == "PSUBUH":
        out = _binary_lanes(a, b, 16, lambda x, y: _sat_unsigned(x - y, 16))
    elif m == "PADDUB":
        out = _binary_lanes(a, b, 8, lambda x, y: _sat_unsigned(x + y, 8))
    elif m == "PSUBUB":
        out = _binary_lanes(a, b, 8, lambda x, y: _sat_unsigned(x - y, 8))
    elif m == "PADSBH":
        lo = _binary_lanes(a, b, 16, lambda x, y: x - y)
        lo &= (1 << 64) - 1
        hi = _binary_lanes(a >> 64, b >> 64, 16, lambda x, y: x + y)
        out = lo | (hi << 64)
    # --- compares / min / max ---
    elif m == "PCGTW":
        out = _compare(a, b, 32, lambda x, y: slane(x, 32) > slane(y, 32))
    elif m == "PCGTH":
        out = _compare(a, b, 16, lambda x, y: slane(x, 16) > slane(y, 16))
    elif m == "PCGTB":
        out = _compare(a, b, 8, lambda x, y: slane(x, 8) > slane(y, 8))
    elif m == "PCEQW":
        out = _compare(a, b, 32, lambda x, y: x == y)
    elif m == "PCEQH":
        out = _compare(a, b, 16, lambda x, y: x == y)
    elif m == "PCEQB":
        out = _compare(a, b, 8, lambda x, y: x == y)
    elif m == "PMAXW":
        out = _binary_lanes(a, b, 32, lambda x, y: x if slane(x, 32) > slane(y, 32) else y)
    elif m == "PMAXH":
        out = _binary_lanes(a, b, 16, lambda x, y: x if slane(x, 16) > slane(y, 16) else y)
    elif m == "PMINW":
        out = _binary_lanes(a, b, 32, lambda x, y: x if slane(x, 32) < slane(y, 32) else y)
    elif m == "PMINH":
        out = _binary_lanes(a, b, 16, lambda x, y: x if slane(x, 16) < slane(y, 16) else y)
    elif m == "PABSW":
        out = _unary_lanes(b, 32, lambda x: _abs_lane(x, 32))
    elif m == "PABSH":
        out = _unary_lanes(b, 16, lambda x: _abs_lane(x, 16))
    # --- interleave / pack ---
    elif m == "PEXTLW":
        rl, tl = lanes(a, 32), lanes(b, 32)
        out = pack([tl[0], rl[0], tl[1], rl[1]], 32)
    elif m == "PEXTUW":
        rl, tl = lanes(a, 32), lanes(b, 32)
        out = pack([tl[2], rl[2], tl[3], rl[3]], 32)
    elif m == "PEXTLH":
        out = _interleave(a, b, 16, 0, 4)
    elif m == "PEXTUH":
        out = _interleave(a, b, 16, 4, 4)
    elif m == "PEXTLB":
        out = _interleave(a, b, 8, 0, 8)
    elif m == "PEXTUB":
        out = _interleave(a, b, 8, 8, 8)
    elif m == "PPACW":
        rl, tl = lanes(a, 32), lanes(b, 32)
        out = pack([tl[0], tl[2], rl[0], rl[2]], 32)
    elif m == "PPACH":
        rl, tl = lanes(a, 16), lanes(b, 16)
        out = pack([tl[0], tl[2], tl[4], tl[6], rl[0], rl[2], rl[4], rl[6]], 16)
    elif m == "PPACB":
        rl, tl = lanes(a, 8), lanes(b, 8)
        out = pack([tl[i] for i in range(0, 16, 2)] + [rl[i] for i in range(0, 16, 2)], 8)
    elif m == "PEXT5":
        def ext(v):
            return (((v & 0x1F) << 3) | ((v & 0x3E0) << 6) |
                    ((v & 0x7C00) << 9) | ((v & 0x8000) << 16))
        out = _unary_lanes(b, 32, ext)
    elif m == "PPAC5":
        def pac(v):
            return (((v >> 3) & 0x1F) | ((v >> 6) & 0x3E0) |
                    ((v >> 9) & 0x7C00) | ((v >> 16) & 0x8000))
        out = _unary_lanes(b, 32, pac)
    # --- permutations / copies ---
    elif m == "PCPYLD":
        out = (b & M64) | ((a & M64) << 64)
    elif m == "PCPYUD":
        out = (a >> 64) | ((b >> 64) << 64)
    elif m == "PCPYH":
        hl = lanes(b, 16)
        out = pack([hl[0]] * 4 + [hl[4]] * 4, 16)
    elif m == "PEXEH":
        hl = lanes(b, 16)
        out = pack([hl[2], hl[1], hl[0], hl[3], hl[6], hl[5], hl[4], hl[7]], 16)
    elif m == "PREVH":
        hl = lanes(b, 16)
        out = pack([hl[3], hl[2], hl[1], hl[0], hl[7], hl[6], hl[5], hl[4]], 16)
    elif m == "PEXCH":
        hl = lanes(b, 16)
        out = pack([hl[0], hl[2], hl[1], hl[3], hl[4], hl[6], hl[5], hl[7]], 16)
    elif m == "PEXEW":
        wl = lanes(b, 32)
        out = pack([wl[2], wl[1], wl[0], wl[3]], 32)
    elif m == "PROT3W":
        wl = lanes(b, 32)
        out = pack([wl[1], wl[2], wl[0], wl[3]], 32)
    elif m == "PEXCW":
        wl = lanes(b, 32)
        out = pack([wl[0], wl[2], wl[1], wl[3]], 32)
    elif m == "PINTEH":
        rl, tl = lanes(a, 16), lanes(b, 16)
        out = pack([tl[0], rl[0], tl[2], rl[2], tl[4], rl[4], tl[6], rl[6]], 16)
    elif m == "PINTH":
        rl, tl = lanes(a, 16), lanes(b, 16)
        out = pack([tl[0], rl[4], tl[1], rl[5], tl[2], rl[6], tl[3], rl[7]], 16)
    # --- 128-bit bitwise ---
    elif m == "PAND":
        out = a & b
    elif m == "POR":
        out = a | b
    elif m == "PXOR":
        out = a ^ b
    elif m == "PNOR":
        out = ~(a | b)
    # --- variable shifts ---
    elif m == "PSLLVW":
        rl, tl = lanes(a, 32), lanes(b, 32)
        out = pack([sext32(tl[0] << (rl[0] & 0x1F)) & M64,
                    sext32(tl[2] << (rl[2] & 0x1F)) & M64], 64)
    elif m == "PSRLVW":
        rl, tl = lanes(a, 32), lanes(b, 32)
        out = pack([sext32(tl[0] >> (rl[0] & 0x1F)) & M64,
                    sext32(tl[2] >> (rl[2] & 0x1F)) & M64], 64)
    elif m == "PSRAVW":
        rl, tl = lanes(a, 32), lanes(b, 32)
        out = pack([s64(sext32(tl[0]) >> (rl[0] & 0x1F)) & M64,
                    s64(sext32(tl[2]) >> (rl[2] & 0x1F)) & M64], 64)
    # --- immediate shifts (sa) ---
    elif m in ("PSLLH", "PSRLH", "PSRAH"):
        out = _shift_lanes(b, 16, insn.shamt & 0xF, m[1:4].lower())
    elif m in ("PSLLW", "PSRLW", "PSRAW"):
        out = _shift_lanes(b, 32, insn.shamt, m[1:4].lower())
    # --- HI/LO helpers ---
    elif m == "PMFHI":
        out = st.hi
    elif m == "PMFLO":
        out = st.lo
    elif m == "PMTHI":
        st.hi = a
        return
    elif m == "PMTLO":
        st.lo = a
        return
    elif m == "PMTHL":
        if insn.shamt == 0:
            al = lanes(a, 32)
            st.lo = (st.lo & ~((M32 << 64) | M32)) | al[0] | (al[2] << 64)
            st.hi = (st.hi & ~((M32 << 64) | M32)) | al[1] | (al[3] << 64)
        return
    # --- MMI multiply/divide side registers (HI1/LO1 = upper 64) ---
    elif m in ("MULT1", "MULTU1"):
        if m == "MULT1":
            prod = slane(st._get(rs) & M32, 32) * slane(st._get(rt) & M32, 32)
        else:
            prod = (st._get(rs) & M32) * (st._get(rt) & M32)
        _set_hi_lo(st, "lo", 1, sext32(prod) & M64)
        _set_hi_lo(st, "hi", 1, sext32(prod >> 32) & M64)
        if rd:
            st._set(rd, sext32(prod))
        return
    elif m in ("DIV1", "DIVU1"):
        dividend = slane(a & M32, 32)
        divisor = slane(b & M32, 32) if m == "DIV1" else (b & M32)
        if divisor == 0:
            lo = 1 if dividend < 0 else -1
            hi = dividend
        elif m == "DIV1" and dividend == -0x80000000 and divisor == -1:
            lo, hi = -0x80000000, 0
        else:
            q = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                q = -q
            lo = q
            hi = dividend - q * divisor
        _set_hi_lo(st, "lo", 1, sext32(lo) & M64)
        _set_hi_lo(st, "hi", 1, sext32(hi) & M64)
        return
    elif m == "MFHI1":
        st._set(rd, sext64((st.hi >> 64) & M64))
        return
    elif m == "MFLO1":
        st._set(rd, sext64((st.lo >> 64) & M64))
        return
    elif m == "MTHI1":
        _set_hi_lo(st, "hi", 1, st._get(rs))
        return
    elif m == "MTLO1":
        _set_hi_lo(st, "lo", 1, st._get(rs))
        return
    elif m == "PLZCW":
        low = _count_leading_sign_bits(lanes(a, 32)[0]) - 1
        high = _count_leading_sign_bits(lanes(a, 32)[1]) - 1
        out = pack([low & M32, high & M32, 0, 0], 32)
    else:
        raise UnsupportedInstruction(insn.addr, insn.raw, m)

    st._set(rd, out)
