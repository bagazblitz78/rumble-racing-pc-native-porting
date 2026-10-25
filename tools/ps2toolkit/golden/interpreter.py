"""Golden-model R5900 interpreter (task T2.3).

Correctness oracle for the recompiler: implements the documented
semantics of the supported instruction subset, including:

- 32-bit GPR semantics (results are 32-bit, $zero hardwired)
- branch/jump delay slots (condition evaluated BEFORE the delay slot;
  link register written when the jump executes)
- big-endian memory, kseg address translation (phys = addr & 0x1FFFFFFF)
- big-endian LWL/LWR/SWL/SWR:
    LWL(A): bytes A..word_end   -> high side of rt   (load pair:  lwl X, lwr X+3)
    LWR(A): bytes word_start..A -> low side of rt
    SWL(A): high side of rt     -> bytes A..word_end (store pair: swl X, swr X+3)
    SWR(A): low side of rt      -> bytes word_start..A
  NOTE: derived semantics — to be cross-validated against an independent
  reference (PCSX2/QEMU) during later milestones (tracked risk).
- R5900 division quirks: DIV by zero -> LO = 0x7FFFFFFF/0x80000000 by
  dividend sign, HI = dividend; INT_MIN/-1 -> LO = INT_MIN, HI = 0.
- no arithmetic overflow traps (R5900 ADD/ADDI/SUB never trap)
"""
from __future__ import annotations

from ..r5900.decoder import DecodedInsn, decode

M32 = 0xFFFFFFFF
RAM_SIZE = 32 * 1024 * 1024
PHYS_MASK = 0x1FFFFFFF


class GuestSyscall(Exception):
    def __init__(self, code: int, pc: int):
        super().__init__(f"syscall code={code:#x} at pc={pc:#010X}")
        self.code = code
        self.pc = pc


class GuestBreak(Exception):
    def __init__(self, code: int, pc: int):
        super().__init__(f"break code={code:#x} at pc={pc:#010X}")
        self.code = code
        self.pc = pc


class MemoryFault(Exception):
    pass


class UnsupportedInstruction(Exception):
    def __init__(self, addr: int, raw: int, mnemonic: str):
        super().__init__(f"unsupported instruction {mnemonic} ({raw:#010x}) at {addr:#010x}")
        self.addr = addr
        self.raw = raw
        self.mnemonic = mnemonic


def s32(value: int) -> int:
    value &= M32
    return value - 0x100000000 if value & 0x80000000 else value


class EEMemory:
    """Flat 32MB PS2 RAM with kseg address translation."""

    def __init__(self, size: int = RAM_SIZE):
        self.size = size
        self.data = bytearray(size)

    def translate(self, addr: int) -> int:
        phys = addr & PHYS_MASK
        if phys >= self.size:
            raise MemoryFault(f"address {addr:#010x} outside RAM (phys {phys:#x})")
        return phys

    def read(self, addr: int, length: int) -> bytes:
        phys = self.translate(addr)
        if phys + length > self.size:
            raise MemoryFault(f"read {addr:#010x}+{length} outside RAM")
        return bytes(self.data[phys : phys + length])

    def write(self, addr: int, blob: bytes):
        phys = self.translate(addr)
        if phys + len(blob) > self.size:
            raise MemoryFault(f"write {addr:#010x}+{len(blob)} outside RAM")
        self.data[phys : phys + len(blob)] = blob

    def read32(self, addr: int) -> int:
        return int.from_bytes(self.read(addr, 4), "big")

    def write32(self, addr: int, value: int):
        self.write(addr, (value & M32).to_bytes(4, "big"))


class EEState:
    def __init__(self, pc: int = 0):
        self.regs = [0] * 32
        self.hi = 0
        self.lo = 0
        self.pc = pc

    def _set(self, reg: int, value: int):
        if reg != 0:
            self.regs[reg] = value & M32

    def _get(self, reg: int) -> int:
        return self.regs[reg]


def _load_be(mem: EEMemory, addr: int, count: int) -> int:
    return int.from_bytes(mem.read(addr, count), "big")


def _exec_alu(st: EEState, mem: EEMemory, insn: DecodedInsn) -> None:
    m = insn.mnemonic
    rs, rt, rd = insn.rs, insn.rt, insn.rd

    if m in ("nop",):
        return
    if m == "addiu":
        st._set(rt, st._get(rs) + insn.imm)
    elif m == "addi":
        st._set(rt, st._get(rs) + insn.imm)
    elif m == "andi":
        st._set(rt, st._get(rs) & insn.uimm)
    elif m == "ori":
        st._set(rt, st._get(rs) | insn.uimm)
    elif m == "xori":
        st._set(rt, st._get(rs) ^ insn.uimm)
    elif m == "lui":
        st._set(rt, insn.uimm << 16)
    elif m == "slti":
        st._set(rt, 1 if s32(st._get(rs)) < insn.imm else 0)
    elif m == "sltiu":
        st._set(rt, 1 if st._get(rs) < ((insn.imm) & M32) else 0)
    elif m in ("sll", "srl", "sra"):
        v = st._get(rt)
        if m == "sll":
            st._set(rd, v << insn.shamt)
        elif m == "srl":
            st._set(rd, v >> insn.shamt)
        else:
            st._set(rd, s32(v) >> insn.shamt)
    elif m in ("sllv", "srlv", "srav"):
        v = st._get(rt)
        sh = st._get(rs) & 0x1F
        if m == "sllv":
            st._set(rd, v << sh)
        elif m == "srlv":
            st._set(rd, v >> sh)
        else:
            st._set(rd, s32(v) >> sh)
    elif m in ("add", "addu"):
        st._set(rd, st._get(rs) + st._get(rt))
    elif m in ("sub", "subu"):
        st._set(rd, st._get(rs) - st._get(rt))
    elif m == "and":
        st._set(rd, st._get(rs) & st._get(rt))
    elif m == "or":
        st._set(rd, st._get(rs) | st._get(rt))
    elif m == "xor":
        st._set(rd, st._get(rs) ^ st._get(rt))
    elif m == "nor":
        st._set(rd, ~(st._get(rs) | st._get(rt)) & M32)
    elif m == "slt":
        st._set(rd, 1 if s32(st._get(rs)) < s32(st._get(rt)) else 0)
    elif m == "sltu":
        st._set(rd, 1 if st._get(rs) < st._get(rt) else 0)
    elif m == "mult":
        prod = s32(st._get(rs)) * s32(st._get(rt))
        st.hi = (prod >> 32) & M32
        st.lo = prod & M32
    elif m == "multu":
        prod = st._get(rs) * st._get(rt)
        st.hi = (prod >> 32) & M32
        st.lo = prod & M32
    elif m == "div":
        dividend, divisor = s32(st._get(rs)), s32(st._get(rt))
        if divisor == 0:
            st.lo = 0x7FFFFFFF if dividend >= 0 else 0x80000000
            st.hi = dividend & M32
        elif dividend == -0x80000000 and divisor == -1:
            st.lo = 0x80000000
            st.hi = 0
        else:
            q = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                q = -q
            r = dividend - q * divisor
            st.lo = q & M32
            st.hi = r & M32
    elif m == "divu":
        dividend, divisor = st._get(rs), st._get(rt)
        if divisor == 0:
            st.lo = M32
            st.hi = dividend
        else:
            st.lo = dividend // divisor
            st.hi = dividend % divisor
    elif m == "mfhi":
        st._set(rd, st.hi)
    elif m == "mflo":
        st._set(rd, st.lo)
    elif m == "mthi":
        st.hi = st._get(rs)
    elif m == "mtlo":
        st.lo = st._get(rs)
    elif m == "syscall":
        raise GuestSyscall((insn.raw >> 6) & 0xFFFFF, insn.addr)
    elif m == "break":
        raise GuestBreak((insn.raw >> 6) & 0xFFFFF, insn.addr)
    elif m in ("lb", "lbu", "lh", "lhu", "lw", "lwl", "lwr"):
        addr = st._get(rs) + insn.imm
        if m == "lb":
            v = mem.read(addr, 1)[0]
            st._set(rt, v - 0x100 if v & 0x80 else v)
        elif m == "lbu":
            st._set(rt, mem.read(addr, 1)[0])
        elif m == "lh":
            v = _load_be(mem, addr, 2)
            st._set(rt, v - 0x10000 if v & 0x8000 else v)
        elif m == "lhu":
            st._set(rt, _load_be(mem, addr, 2))
        elif m == "lw":
            st._set(rt, _load_be(mem, addr, 4))
        elif m == "lwl":
            byte = addr & 3
            n = 4 - byte
            val = _load_be(mem, addr, n)
            keep = (1 << (8 * byte)) - 1  # keep low `byte` bytes of rt
            st._set(rt, (val << (8 * byte)) | (st._get(rt) & keep))
        elif m == "lwr":
            byte = addr & 3
            n = byte + 1
            aligned = addr & ~3
            val = _load_be(mem, aligned, n)
            keep = M32 ^ ((1 << (8 * n)) - 1)  # keep high (4-n) bytes of rt
            st._set(rt, (st._get(rt) & keep) | val)
    elif m in ("sb", "sh", "sw", "swl", "swr"):
        addr = st._get(rs) + insn.imm
        value = st._get(rt)
        if m == "sb":
            mem.write(addr, bytes([value & 0xFF]))
        elif m == "sh":
            mem.write(addr, (value & 0xFFFF).to_bytes(2, "big"))
        elif m == "sw":
            mem.write(addr, value.to_bytes(4, "big"))
        elif m == "swl":
            byte = addr & 3
            n = 4 - byte
            part = value >> (8 * byte)  # high `n` bytes of rt
            mem.write(addr, part.to_bytes(n, "big"))
        elif m == "swr":
            byte = addr & 3
            n = byte + 1
            aligned = addr & ~3
            part = value & ((1 << (8 * n)) - 1)  # low `n` bytes of rt
            mem.write(aligned, part.to_bytes(n, "big"))
    else:
        raise UnsupportedInstruction(insn.addr, insn.raw, insn.mnemonic)


def _branch_target(st: EEState, insn: DecodedInsn) -> tuple[bool, int]:
    """Evaluate branch/jump condition with pre-delay-slot state."""
    m = insn.mnemonic
    rs, rt = insn.rs, insn.rt
    if m == "beq":
        return st._get(rs) == st._get(rt), insn.addr + 4 + insn.imm * 4
    if m == "bne":
        return st._get(rs) != st._get(rt), insn.addr + 4 + insn.imm * 4
    if m == "blez":
        return s32(st._get(rs)) <= 0, insn.addr + 4 + insn.imm * 4
    if m == "bgtz":
        return s32(st._get(rs)) > 0, insn.addr + 4 + insn.imm * 4
    if m == "bltz":
        return s32(st._get(rs)) < 0, insn.addr + 4 + insn.imm * 4
    if m == "bgez":
        return s32(st._get(rs)) >= 0, insn.addr + 4 + insn.imm * 4
    if m == "j":
        return True, insn.target
    if m == "jal":
        return True, insn.target
    if m == "jr":
        return True, st._get(rs) & 0xFFFFFFFC
    if m == "jalr":
        return True, st._get(rs) & 0xFFFFFFFC
    raise UnsupportedInstruction(insn.addr, insn.raw, insn.mnemonic)


def step(st: EEState, mem: EEMemory) -> int:
    """Execute one instruction (plus its delay slot if any).

    Returns the number of guest instructions executed (1 or 2).
    """
    insn = decode(mem.read32(st.pc), st.pc)
    if insn.mnemonic == "unsupported":
        raise UnsupportedInstruction(st.pc, insn.raw, f"op={insn.opcode:#x}")

    if insn.has_delay_slot:
        taken, target = _branch_target(st, insn)  # condition: pre-delay state
        # link register is written when the jump executes (before delay slot)
        if insn.is_link:
            if insn.mnemonic == "jal":
                st._set(31, insn.addr + 8)
            else:  # jalr
                st._set(insn.rd, insn.addr + 8)
        delay = decode(mem.read32(st.pc + 4), st.pc + 4)
        if delay.has_delay_slot:
            raise UnsupportedInstruction(
                delay.addr, delay.raw, "branch in delay slot (illegal)"
            )
        if delay.mnemonic == "unsupported":
            raise UnsupportedInstruction(delay.addr, delay.raw, f"op={delay.opcode:#x}")
        _exec_alu(st, mem, delay)  # delay slot executes unconditionally
        st.pc = target if taken else insn.addr + 8
        return 2

    _exec_alu(st, mem, insn)
    st.pc = insn.addr + 4
    return 1


def run(st: EEState, mem: EEMemory, max_steps: int, until: int | None = None) -> int:
    """Run until pc == until or max_steps reached. Returns steps executed."""
    steps = 0
    while steps < max_steps:
        if until is not None and st.pc == until:
            break
        steps += step(st, mem)
    return steps
