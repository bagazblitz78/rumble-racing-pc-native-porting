"""Minimal x64 machine-code encoder (task T2.5).

Encodes exactly the instruction forms needed by the PS2->x64 code
generator, with label/rel32 fixup support. Register numbers follow the
x86-64 encoding order (RAX=0 ... R15=15). All memory-slot helpers assume
the Win64 context-pointer scheme used by the emitter (r15 = guest state,
r14 = guest RAM base).
"""
from __future__ import annotations

import struct

RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = range(8)
R8, R9, R10, R11, R12, R13, R14, R15 = range(8, 16)

_CC = {
    "o": 0x0, "no": 0x1, "b": 0x2, "c": 0x2, "ae": 0x3, "nc": 0x3,
    "e": 0x4, "z": 0x4, "ne": 0x5, "nz": 0x5,
    "be": 0x6, "a": 0x7, "s": 0x8, "ns": 0x9,
    "p": 0xA, "np": 0xB, "l": 0xC, "ge": 0xD, "le": 0xE, "g": 0xF,
}

_ALU_OP = {"add": 0x01, "or": 0x09, "and": 0x21, "sub": 0x29, "xor": 0x31, "cmp": 0x39}
_ALU_EXT = {"add": 0, "or": 1, "and": 4, "sub": 5, "xor": 6, "cmp": 7}
_SHIFT_EXT = {"shl": 4, "shr": 5, "sar": 7}


def _rex(w: bool = False, r: int = 0, x: int = 0, b: int = 0) -> int | None:
    """REX prefix from full register indices (0-15): bits are reg >= 8.

    Returns None when no REX byte is required.
    """
    value = (
        0x40
        | (int(w) << 3)
        | (((r >> 3) & 1) << 2)
        | (((x >> 3) & 1) << 1)
        | ((b >> 3) & 1)
    )
    return None if value == 0x40 else value


def _modrm(mod: int, reg: int, rm: int) -> int:
    return (mod << 6) | ((reg & 7) << 3) | (rm & 7)


class Label:
    __slots__ = ("offset",)

    def __init__(self):
        self.offset: int | None = None


class Asm:
    """Byte buffer for emitted machine code with label fixups."""

    def __init__(self):
        self.buf = bytearray()
        self._fixups: list[tuple[int, Label]] = []  # (rel32 position, label)

    def __len__(self):
        return len(self.buf)

    def emit(self, *parts: int | bytes | None):
        for part in parts:
            if part is None:
                continue
            if isinstance(part, int):
                self.buf.append(part & 0xFF)
            else:
                self.buf += part

    # --- labels / control flow ---

    def new_label(self) -> Label:
        return Label()

    def bind(self, label: Label):
        label.offset = len(self.buf)
        remaining = []
        for pos, lbl in self._fixups:
            if lbl is label:
                rel = label.offset - (pos + 4)
                struct.pack_into("<i", self.buf, pos, rel)
            else:
                remaining.append((pos, lbl))
        self._fixups = remaining

    def _rel32(self, label: Label):
        if label.offset is not None:
            rel = label.offset - (len(self.buf) + 4)
            self.emit(struct.pack("<i", rel))
        else:
            self._fixups.append((len(self.buf), label))
            self.emit(struct.pack("<i", 0))

    def jmp(self, label: Label):
        self.emit(0xE9)
        self._rel32(label)

    def jcc(self, cc: str, label: Label):
        self.emit(0x0F, 0x80 + _CC[cc])
        self._rel32(label)

    def call(self, label: Label):
        self.emit(0xE8)
        self._rel32(label)

    def ret(self):
        self.emit(0xC3)

    def nop(self):
        self.emit(0x90)

    # --- moves ---

    def mov_ri(self, dst: int, imm: int):
        """mov r32, imm32 (zero-extends into the 64-bit register)."""
        if dst >= 8:
            self.emit(0x41)
        self.emit(0xB8 + (dst & 7), struct.pack("<I", imm & 0xFFFFFFFF))

    def mov_rr(self, dst: int, src: int):
        """mov r64, r64 (64-bit)."""
        self.emit(_rex(w=True, r=src, b=dst), 0x89, _modrm(3, src, dst))

    def ld32(self, dst: int, base: int, disp: int = 0):
        """mov r32, dword ptr [base + disp32] (zero-extends)."""
        self.emit(_rex(r=dst, b=base), 0x8B, _modrm(2, dst, base), struct.pack("<i", disp))

    def st32(self, base: int, disp: int, src: int):
        """mov dword ptr [base + disp32], r32."""
        self.emit(_rex(r=src, b=base), 0x89, _modrm(2, src, base), struct.pack("<i", disp))

    def st64(self, base: int, disp: int, src: int):
        """mov qword ptr [base + disp32], r64."""
        self.emit(_rex(w=True, r=src, b=base), 0x89, _modrm(2, src, base),
                  struct.pack("<i", disp))

    def ld32_ram(self, dst: int, base: int, idx: int):
        """mov r32, dword ptr [base + idx] (byte address in idx)."""
        self.emit(_rex(r=dst, x=idx, b=base), 0x8B, _modrm(0, dst, 4), _modrm(0, idx, base))

    def st32_ram(self, base: int, idx: int, src: int):
        """mov dword ptr [base + idx], r32."""
        self.emit(_rex(r=src, x=idx, b=base), 0x89, _modrm(0, src, 4), _modrm(0, idx, base))

    def ld64_ram(self, dst: int, base: int, idx: int):
        """mov r64, qword ptr [base + idx] (byte address in idx)."""
        self.emit(_rex(w=True, r=dst, x=idx, b=base), 0x8B, _modrm(0, dst, 4),
                  _modrm(0, idx, base))

    def st64_ram(self, base: int, idx: int, src: int):
        """mov qword ptr [base + idx], r64."""
        self.emit(_rex(w=True, r=src, x=idx, b=base), 0x89, _modrm(0, src, 4),
                  _modrm(0, idx, base))

    def ld64_rm(self, dst: int, base: int, disp: int = 0):
        """mov r64, qword ptr [base + disp32]."""
        self.emit(_rex(w=True, r=dst, b=base), 0x8B, _modrm(2, dst, base),
                  struct.pack("<i", disp))

    def ld8_ram(self, dst: int, base: int, idx: int):
        """mov r8, byte ptr [base + idx]."""
        self.emit(_rex(r=dst, x=idx, b=base), 0x8A, _modrm(0, dst, 4), _modrm(0, idx, base))

    def ld16_ram(self, dst: int, base: int, idx: int):
        """mov r16, word ptr [base + idx]."""
        self.emit(0x66, _rex(r=dst, x=idx, b=base), 0x8B, _modrm(0, dst, 4),
                  _modrm(0, idx, base))

    def st8_ram(self, base: int, idx: int, src: int):
        """mov byte ptr [base + idx], r8."""
        self.emit(_rex(r=src, x=idx, b=base), 0x88, _modrm(0, src, 4), _modrm(0, idx, base))

    def st16_ram(self, base: int, idx: int, src: int):
        """mov word ptr [base + idx], r16."""
        self.emit(0x66, _rex(r=src, x=idx, b=base), 0x89, _modrm(0, src, 4),
                  _modrm(0, idx, base))

    def movsx8(self, dst: int, src8: int):
        """movsx r32, r8."""
        self.emit(_rex(r=dst, b=src8), 0x0F, 0xBE, _modrm(3, dst, src8))

    def movsx16(self, dst: int, src16: int):
        """movsx r32, r16 (opcode 0F BF)."""
        self.emit(_rex(r=dst, b=src16), 0x0F, 0xBF, _modrm(3, dst, src16))

    def xchg_al_ah(self):
        """xchg al, ah (swap the two low bytes of AX; unused since LE data)."""
        self.emit(0x86, 0xC4)

    def mov_ri64(self, dst: int, imm: int):
        """movabs r64, imm64."""
        self.emit(_rex(w=True, b=dst), 0xB8 + (dst & 7),
                  struct.pack("<Q", imm & 0xFFFFFFFFFFFFFFFF))

    def jmp_mem_rip(self, disp: int):
        """jmp qword ptr [rip + disp32] (indirect jump via memory, e.g. IAT)."""
        self.emit(0xFF, 0x25, struct.pack("<i", disp))

    def movzx(self, dst: int, src8: int):
        self.emit(_rex(r=dst, b=src8), 0x0F, 0xB6, _modrm(3, dst, src8))

    def movzx16(self, dst: int, src16: int):
        self.emit(_rex(r=dst, b=src16), 0x0F, 0xB7, _modrm(3, dst, src16))

    def movsxd(self, dst: int, src32: int):
        self.emit(_rex(w=True, r=dst, b=src32), 0x63, _modrm(3, dst, src32))

    # --- arithmetic ---

    def alu_rr(self, op: str, dst: int, src: int):
        """op r/m32, r32 (dst := dst op src)."""
        self.emit(_rex(r=src, b=dst), _ALU_OP[op], _modrm(3, src, dst))

    def alu_rr64(self, op: str, dst: int, src: int):
        """op r/m64, r64 (dst := dst op src)."""
        self.emit(_rex(w=True, r=src, b=dst), _ALU_OP[op], _modrm(3, src, dst))

    def alu_ri(self, op: str, dst: int, imm: int):
        """op r/m32, imm32."""
        self.emit(_rex(b=dst), 0x81, _modrm(3, _ALU_EXT[op], dst),
                  struct.pack("<i", imm if imm < 0x80000000 else imm - 0x100000000))

    def alu_ri64(self, op: str, dst: int, imm: int):
        """op r/m64, imm32 (sign-extended immediate)."""
        self.emit(_rex(w=True, b=dst), 0x81, _modrm(3, _ALU_EXT[op], dst),
                  struct.pack("<i", imm if imm < 0x80000000 else imm - 0x100000000))

    def shift_ri(self, op: str, dst: int, amount: int):
        self.emit(_rex(b=dst), 0xC1, _modrm(3, _SHIFT_EXT[op], dst), amount & 0x1F)

    def shift_ri64(self, op: str, dst: int, amount: int):
        self.emit(_rex(w=True, b=dst), 0xC1, _modrm(3, _SHIFT_EXT[op], dst),
                  amount & 0x3F)

    def shift_cl(self, op: str, dst: int):
        self.emit(_rex(b=dst), 0xD3, _modrm(3, _SHIFT_EXT[op], dst))

    def shift_cl64(self, op: str, dst: int):
        self.emit(_rex(w=True, b=dst), 0xD3, _modrm(3, _SHIFT_EXT[op], dst))

    def setcc(self, cc: str, dst: int):
        """setcc r8."""
        self.emit(_rex(b=dst), 0x0F, 0x90 + _CC[cc], _modrm(3, 0, dst))

    def test_rr(self, a: int, b: int):
        self.emit(_rex(r=b, b=a), 0x85, _modrm(3, b, a))

    def test_rr64(self, a: int, b: int):
        self.emit(_rex(w=True, r=b, b=a), 0x85, _modrm(3, b, a))

    def bswap(self, reg: int):
        self.emit(_rex(b=reg), 0x0F, 0xC8 + (reg & 7))

    def bswap64(self, reg: int):
        self.emit(_rex(w=True, b=reg), 0x0F, 0xC8 + (reg & 7))

    def imul_rr(self, dst: int, src: int):
        self.emit(_rex(r=dst, b=src), 0x0F, 0xAF, _modrm(3, dst, src))

    def imul64_rr(self, dst: int, src: int):
        self.emit(_rex(w=True, r=dst, b=src), 0x0F, 0xAF, _modrm(3, dst, src))

    def bsr(self, dst: int, src: int):
        """bsr dst, src (index of highest set bit; src must be non-zero)."""
        self.emit(_rex(r=dst, b=src), 0x0F, 0xBD, _modrm(3, dst, src))

    def not_r(self, reg: int):
        self.emit(_rex(b=reg), 0xF7, _modrm(3, 2, reg))

    def not_r64(self, reg: int):
        self.emit(_rex(w=True, b=reg), 0xF7, _modrm(3, 2, reg))

    def shl64_cl(self, reg: int):
        self.emit(_rex(w=True, b=reg), 0xD3, _modrm(3, 4, reg))

    def shr64_ri(self, reg: int, amount: int):
        self.emit(_rex(w=True, b=reg), 0xC1, _modrm(3, 5, reg), amount & 0x3F)

    def shl64_ri(self, reg: int, amount: int):
        self.emit(_rex(w=True, b=reg), 0xC1, _modrm(3, 4, reg), amount & 0x3F)

    def mul_r(self, src: int):
        self.emit(_rex(b=src), 0xF7, _modrm(3, 4, src))

    def cdq(self):
        self.emit(0x99)

    def idiv_r(self, src: int):
        self.emit(_rex(b=src), 0xF7, _modrm(3, 7, src))

    def div_r(self, src: int):
        self.emit(_rex(b=src), 0xF7, _modrm(3, 6, src))

    def neg_r(self, reg: int):
        self.emit(_rex(b=reg), 0xF7, _modrm(3, 3, reg))

    # --- stack / misc ---

    def push(self, reg: int):
        if reg >= 8:
            self.emit(0x41)
        self.emit(0x50 + (reg & 7))

    def pop(self, reg: int):
        if reg >= 8:
            self.emit(0x41)
        self.emit(0x58 + (reg & 7))

    # --- SSE2 (scratch registers xmm0-xmm3; all caller-saved) ---

    def _xmm_rr(self, prefix: int, opcode: int, dst: int, src: int):
        if prefix:
            self.emit(prefix)
        self.emit(_rex(r=dst, b=src), 0x0F, opcode, _modrm(3, dst, src))

    def _xmm_rm(self, prefix: int, opcode: int, dst: int, base: int, disp: int):
        if prefix:
            self.emit(prefix)
        self.emit(_rex(r=dst, b=base), 0x0F, opcode,
                  _modrm(2, dst, base), struct.pack("<i", disp))

    def _xmm_mr(self, prefix: int, opcode: int, base: int, disp: int, src: int):
        if prefix:
            self.emit(prefix)
        self.emit(_rex(r=src, b=base), 0x0F, opcode,
                  _modrm(2, src, base), struct.pack("<i", disp))

    def movdqu_ld(self, dst: int, base: int, disp: int):
        """movdqu xmm, m128."""
        self._xmm_rm(0xF3, 0x6F, dst, base, disp)

    def movdqu_st(self, base: int, disp: int, src: int):
        """movdqu m128, xmm."""
        self._xmm_mr(0xF3, 0x7F, base, disp, src)

    def movdqu_ld_idx(self, dst: int, base: int, idx: int):
        """movdqu xmm, [base + idx] (byte address in idx)."""
        self.emit(0xF3, _rex(r=dst, x=idx, b=base), 0x0F, 0x6F,
                  _modrm(0, dst, 4), _modrm(0, idx, base))

    def movdqu_st_idx(self, base: int, idx: int, src: int):
        """movdqu [base + idx], xmm."""
        self.emit(0xF3, _rex(r=src, x=idx, b=base), 0x0F, 0x7F,
                  _modrm(0, src, 4), _modrm(0, idx, base))

    def paddb(self, dst, src): self._xmm_rr(0x66, 0xFC, dst, src)
    def paddw(self, dst, src): self._xmm_rr(0x66, 0xFD, dst, src)
    def paddd(self, dst, src): self._xmm_rr(0x66, 0xFE, dst, src)
    def psubb(self, dst, src): self._xmm_rr(0x66, 0xF8, dst, src)
    def psubw(self, dst, src): self._xmm_rr(0x66, 0xF9, dst, src)
    def psubd(self, dst, src): self._xmm_rr(0x66, 0xFA, dst, src)
    def paddusb(self, dst, src): self._xmm_rr(0x66, 0xDC, dst, src)
    def paddusw(self, dst, src): self._xmm_rr(0x66, 0xDD, dst, src)
    def psubusb(self, dst, src): self._xmm_rr(0x66, 0xD8, dst, src)
    def psubusw(self, dst, src): self._xmm_rr(0x66, 0xD9, dst, src)
    def paddsb(self, dst, src): self._xmm_rr(0x66, 0xEC, dst, src)
    def paddsw(self, dst, src): self._xmm_rr(0x66, 0xED, dst, src)
    def psubsb(self, dst, src): self._xmm_rr(0x66, 0xE8, dst, src)
    def psubsw(self, dst, src): self._xmm_rr(0x66, 0xE9, dst, src)
    def pcmpeqb(self, dst, src): self._xmm_rr(0x66, 0x74, dst, src)
    def pcmpeqw(self, dst, src): self._xmm_rr(0x66, 0x75, dst, src)
    def pcmpeqd(self, dst, src): self._xmm_rr(0x66, 0x76, dst, src)
    def pcmpgtb(self, dst, src): self._xmm_rr(0x66, 0x64, dst, src)
    def pcmpgtw(self, dst, src): self._xmm_rr(0x66, 0x65, dst, src)
    def pcmpgtd(self, dst, src): self._xmm_rr(0x66, 0x66, dst, src)
    def pmaxsw(self, dst, src): self._xmm_rr(0x66, 0xEE, dst, src)
    def pminsw(self, dst, src): self._xmm_rr(0x66, 0xEA, dst, src)
    def pand(self, dst, src): self._xmm_rr(0x66, 0xDB, dst, src)
    def pandn(self, dst, src): self._xmm_rr(0x66, 0xDF, dst, src)
    def por(self, dst, src): self._xmm_rr(0x66, 0xEB, dst, src)
    def pxor(self, dst, src): self._xmm_rr(0x66, 0xEF, dst, src)
    def punpcklbw(self, dst, src): self._xmm_rr(0x66, 0x60, dst, src)
    def punpcklwd(self, dst, src): self._xmm_rr(0x66, 0x61, dst, src)
    def punpckldq(self, dst, src): self._xmm_rr(0x66, 0x62, dst, src)
    def punpckhbw(self, dst, src): self._xmm_rr(0x66, 0x68, dst, src)
    def punpckhwd(self, dst, src): self._xmm_rr(0x66, 0x69, dst, src)
    def punpckhdq(self, dst, src): self._xmm_rr(0x66, 0x6A, dst, src)
    def punpcklqdq(self, dst, src): self._xmm_rr(0x66, 0x6C, dst, src)
    def punpckhqdq(self, dst, src): self._xmm_rr(0x66, 0x6D, dst, src)

    def pshuflw(self, dst, src, imm):
        self.emit(0xF2, _rex(r=dst, b=src), 0x0F, 0x70,
                  _modrm(3, dst, src), imm & 0xFF)

    def pshufhw(self, dst, src, imm):
        self.emit(0xF3, _rex(r=dst, b=src), 0x0F, 0x70,
                  _modrm(3, dst, src), imm & 0xFF)

    def pshufd(self, dst, src, imm):
        self.emit(0x66, _rex(r=dst, b=src), 0x0F, 0x70,
                  _modrm(3, dst, src), imm & 0xFF)

    def pmovmskb(self, dst: int, src: int):
        self.emit(0x66, _rex(r=dst, b=src), 0x0F, 0xD7,
                  _modrm(3, dst, src))

    # --- scalar single-precision FPU ---

    def movd_ld(self, dst: int, base: int, disp: int):
        """movd xmm, dword ptr [base+disp] (zero-extends 32 bits)."""
        self._xmm_rm(0x66, 0x6E, dst, base, disp)

    def movd_st(self, base: int, disp: int, src: int):
        """movd dword ptr [base+disp], xmm."""
        self._xmm_mr(0x66, 0x7E, base, disp, src)

    def movd_xmm_gpr(self, dst: int, src: int):
        """movd xmm, r32."""
        self._xmm_rr(0x66, 0x6E, dst, src)

    def movd_gpr_xmm(self, dst: int, src: int):
        """movd r32, xmm."""
        self._xmm_rr(0x66, 0x7E, dst, src)

    def movss_ld(self, dst: int, base: int, disp: int):
        self._xmm_rm(0xF3, 0x10, dst, base, disp)

    def movss_st(self, base: int, disp: int, src: int):
        self._xmm_mr(0xF3, 0x11, base, disp, src)

    def addss(self, dst, src): self._xmm_rr(0xF3, 0x58, dst, src)
    def subss(self, dst, src): self._xmm_rr(0xF3, 0x5C, dst, src)
    def mulss(self, dst, src): self._xmm_rr(0xF3, 0x59, dst, src)
    def divss(self, dst, src): self._xmm_rr(0xF3, 0x5E, dst, src)
    def sqrtss(self, dst, src): self._xmm_rr(0xF3, 0x51, dst, src)
    def minss(self, dst, src): self._xmm_rr(0xF3, 0x5D, dst, src)
    def maxss(self, dst, src): self._xmm_rr(0xF3, 0x5F, dst, src)
    def ucomiss(self, dst, src): self._xmm_rr(0x00, 0x2E, dst, src)
    def xorps(self, dst, src): self._xmm_rr(0x00, 0x57, dst, src)
    def andps(self, dst, src): self._xmm_rr(0x00, 0x54, dst, src)
    def orps(self, dst, src): self._xmm_rr(0x00, 0x56, dst, src)
    def cvtsi2ss(self, dst, src): self._xmm_rr(0xF3, 0x2A, dst, src)
    def cvttss2si(self, dst, src): self._xmm_rr(0xF3, 0x2C, dst, src)

    def int3(self):
        self.emit(0xCC)

    def finalize(self) -> bytes:
        if self._fixups:
            unbound = ", ".join(str(lbl.offset) for _, lbl in self._fixups)
            raise RuntimeError(f"unbound labels remain: {len(self._fixups)} ({unbound})")
        return bytes(self.buf)
