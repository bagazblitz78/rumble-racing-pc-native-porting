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
M64 = 0xFFFFFFFFFFFFFFFF
M128 = (1 << 128) - 1
RAM_SIZE = 32 * 1024 * 1024
PHYS_MASK = 0x1FFFFFFF


class GuestSyscall(Exception):
    def __init__(self, code: int, pc: int, number: int | None = None):
        super().__init__(f"syscall code={code:#x} at pc={pc:#010X}")
        self.code = code          # instruction code field (used by tests/PE)
        self.pc = pc
        # PS2 SDK code uses `syscall 0` and passes the kernel function number
        # in $v1; the boot tracer dispatches on this when code == 0.
        self.number = code if number is None else number


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


def s64(value: int) -> int:
    value &= M64
    return value - 0x10000000000000000 if value & 0x8000000000000000 else value


def sext32(value: int) -> int:
    """Sign-extend a 32-bit result to the full 128-bit GPR width.

    Every 32-bit operation writes a sign-extended result; the scalar core
    only ever reads the low 64 bits, but LQ/SQ and (future) MMI observe the
    upper half.
    """
    value &= M32
    return (value | (M128 ^ M32)) if value & 0x80000000 else value


def sext64(value: int) -> int:
    """Sign-extend a 64-bit result to the full 128-bit GPR width."""
    value &= M64
    return (value | (M128 ^ M64)) if value & 0x8000000000000000 else value


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
        """Big-endian data word (the EE's data endianness)."""
        return int.from_bytes(self.read(addr, 4), "big")

    def fetch32(self, addr: int) -> int:
        """Little-endian instruction word (MIPS instruction fetch)."""
        return int.from_bytes(self.read(addr, 4), "little")

    def write32(self, addr: int, value: int):
        self.write(addr, (value & M32).to_bytes(4, "big"))


class EEState:
    def __init__(self, pc: int = 0):
        self.regs = [0] * 32
        self.hi = 0
        self.lo = 0
        self.cop0 = [0] * 32  # COP0 data/control registers (32-bit)
        self.pc = pc

    def _set(self, reg: int, value: int):
        if reg != 0:
            self.regs[reg] = value & M128

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
        st._set(rt, sext32(st._get(rs) + insn.imm))
    elif m == "addi":
        st._set(rt, sext32(st._get(rs) + insn.imm))
    elif m == "daddiu":
        st._set(rt, sext64(st._get(rs) + insn.imm))
    elif m == "daddi":
        st._set(rt, sext64(st._get(rs) + insn.imm))
    elif m == "andi":
        st._set(rt, st._get(rs) & insn.uimm)
    elif m == "ori":
        st._set(rt, st._get(rs) | insn.uimm)
    elif m == "xori":
        st._set(rt, st._get(rs) ^ insn.uimm)
    elif m == "lui":
        st._set(rt, sext32(insn.uimm << 16))
    elif m == "slti":
        st._set(rt, 1 if s64(st._get(rs)) < insn.imm else 0)
    elif m == "sltiu":
        st._set(rt, 1 if (st._get(rs) & M64) < (insn.imm & M64) else 0)
    elif m in ("sll", "srl", "sra"):
        v = st._get(rt) & M32
        if m == "sll":
            st._set(rd, sext32(v << insn.shamt))
        elif m == "srl":
            st._set(rd, sext32(v >> insn.shamt))
        else:
            st._set(rd, sext32(s32(v) >> insn.shamt))
    elif m in ("sllv", "srlv", "srav"):
        v = st._get(rt) & M32
        sh = st._get(rs) & 0x1F
        if m == "sllv":
            st._set(rd, sext32(v << sh))
        elif m == "srlv":
            st._set(rd, sext32(v >> sh))
        else:
            st._set(rd, sext32(s32(v) >> sh))
    elif m in ("dsll", "dsrl", "dsra", "dsll32", "dsrl32", "dsra32"):
        amount = insn.shamt + (32 if m.endswith("32") else 0)
        v = st._get(rt) & M64
        if m.startswith("dsll"):
            st._set(rd, sext64(v << amount))
        elif m.startswith("dsrl"):
            st._set(rd, sext64(v >> amount))
        else:
            st._set(rd, sext64(s64(v) >> amount))
    elif m in ("dsllv", "dsrlv", "dsrav"):
        v = st._get(rt) & M64
        sh = st._get(rs) & 0x3F
        if m == "dsllv":
            st._set(rd, sext64(v << sh))
        elif m == "dsrlv":
            st._set(rd, sext64(v >> sh))
        else:
            st._set(rd, sext64(s64(v) >> sh))
    elif m == "movz":
        if st._get(rt) == 0:
            st._set(rd, st._get(rs))
    elif m == "movn":
        if st._get(rt) != 0:
            st._set(rd, st._get(rs))
    elif m in ("sync", "cache", "pref"):
        return  # no caches / coherence model in the single-threaded golden core
    elif m == "mfc0":
        st._set(rt, sext32(st.cop0[insn.rd]))
    elif m == "mtc0":
        st.cop0[insn.rd] = st._get(rt) & M32
    elif m == "ei":
        st.cop0[12] |= 1          # Status.IEc
    elif m == "di":
        st.cop0[12] &= ~1
    elif m == "eret":
        st.cop0[12] &= ~2         # clear Status.EXL
    elif m in ("add", "addu"):
        st._set(rd, sext32(st._get(rs) + st._get(rt)))
    elif m in ("sub", "subu"):
        st._set(rd, sext32(st._get(rs) - st._get(rt)))
    elif m == "and":
        st._set(rd, st._get(rs) & st._get(rt))
    elif m == "or":
        st._set(rd, st._get(rs) | st._get(rt))
    elif m == "xor":
        st._set(rd, st._get(rs) ^ st._get(rt))
    elif m == "nor":
        st._set(rd, ~(st._get(rs) | st._get(rt)))
    elif m == "slt":
        st._set(rd, 1 if s64(st._get(rs)) < s64(st._get(rt)) else 0)
    elif m == "sltu":
        st._set(rd, 1 if (st._get(rs) & M64) < (st._get(rt) & M64) else 0)
    elif m in ("dadd", "daddu"):
        st._set(rd, sext64(st._get(rs) + st._get(rt)))
    elif m in ("dsub", "dsubu"):
        st._set(rd, sext64(st._get(rs) - st._get(rt)))
    elif m == "mult":
        prod = s32(st._get(rs)) * s32(st._get(rt))
        st.hi = (prod >> 32) & M32
        st.lo = prod & M32
    elif m == "multu":
        prod = (st._get(rs) & M32) * (st._get(rt) & M32)
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
        dividend, divisor = st._get(rs) & M32, st._get(rt) & M32
        if divisor == 0:
            st.lo = M32
            st.hi = dividend
        else:
            st.lo = dividend // divisor
            st.hi = dividend % divisor
    elif m == "mfhi":
        st._set(rd, sext32(st.hi))
    elif m == "mflo":
        st._set(rd, sext32(st.lo))
    elif m == "mthi":
        st.hi = st._get(rs) & M32
    elif m == "mtlo":
        st.lo = st._get(rs) & M32
    elif m == "syscall":
        raise GuestSyscall((insn.raw >> 6) & 0xFFFFF, insn.addr,
                           number=st._get(3))
    elif m == "break":
        raise GuestBreak((insn.raw >> 6) & 0xFFFFF, insn.addr)
    elif m in ("lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "lq", "lwl", "lwr"):
        addr = (st._get(rs) + insn.imm) & M32
        if m == "lb":
            v = mem.read(addr, 1)[0]
            st._set(rt, sext32(v - 0x100 if v & 0x80 else v))
        elif m == "lbu":
            st._set(rt, mem.read(addr, 1)[0])
        elif m == "lh":
            v = _load_be(mem, addr, 2)
            st._set(rt, sext32(v - 0x10000 if v & 0x8000 else v))
        elif m == "lhu":
            st._set(rt, _load_be(mem, addr, 2))
        elif m == "lw":
            st._set(rt, sext32(_load_be(mem, addr, 4)))
        elif m == "lwu":
            st._set(rt, _load_be(mem, addr, 4))
        elif m == "ld":
            st._set(rt, sext64(_load_be(mem, addr, 8)))
        elif m == "lq":
            st._set(rt, _load_be(mem, addr, 16))
        elif m == "lwl":
            byte = addr & 3
            n = 4 - byte
            val = _load_be(mem, addr, n)
            keep = (1 << (8 * byte)) - 1  # keep low `byte` bytes of rt
            st._set(rt, sext32((val << (8 * byte)) | (st._get(rt) & keep)))
        elif m == "lwr":
            byte = addr & 3
            n = byte + 1
            aligned = addr & ~3
            val = _load_be(mem, aligned, n)
            keep = M32 ^ ((1 << (8 * n)) - 1)  # keep high (4-n) bytes of rt
            st._set(rt, sext32((st._get(rt) & keep) | val))
    elif m in ("sb", "sh", "sw", "sd", "sq", "swl", "swr"):
        addr = (st._get(rs) + insn.imm) & M32
        value = st._get(rt)
        if m == "sb":
            mem.write(addr, bytes([value & 0xFF]))
        elif m == "sh":
            mem.write(addr, (value & 0xFFFF).to_bytes(2, "big"))
        elif m == "sw":
            mem.write(addr, (value & M32).to_bytes(4, "big"))
        elif m == "sd":
            mem.write(addr, (value & M64).to_bytes(8, "big"))
        elif m == "sq":
            mem.write(addr, (value & M128).to_bytes(16, "big"))
        elif m == "swl":
            byte = addr & 3
            n = 4 - byte
            part = (value & M32) >> (8 * byte)  # high `n` bytes of rt
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
    target = insn.addr + 4 + insn.imm * 4
    if m in ("beq", "beql"):
        return (st._get(rs) & M64) == (st._get(rt) & M64), target
    if m in ("bne", "bnel"):
        return (st._get(rs) & M64) != (st._get(rt) & M64), target
    if m in ("blez", "blezl"):
        return s64(st._get(rs)) <= 0, target
    if m in ("bgtz", "bgtzl"):
        return s64(st._get(rs)) > 0, target
    if m in ("bltz", "bltzl", "bltzal", "bltzall"):
        return s64(st._get(rs)) < 0, target
    if m in ("bgez", "bgezl", "bgezal", "bgezall"):
        return s64(st._get(rs)) >= 0, target
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
    insn = decode(mem.fetch32(st.pc), st.pc)
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
        elif insn.is_branch_link and taken:
            st._set(31, insn.addr + 8)
        if insn.is_likely and not taken:
            # branch-likely: a not-taken branch nullifies its delay slot
            st.pc = insn.addr + 8
            return 1
        delay = decode(mem.fetch32(st.pc + 4), st.pc + 4)
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
    if insn.mnemonic == "eret":
        st.pc = st.cop0[14] & M32  # return to EPC
    else:
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
