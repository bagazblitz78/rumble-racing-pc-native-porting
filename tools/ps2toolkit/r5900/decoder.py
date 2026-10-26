"""MIPS R5900 (Emotion Engine) instruction decoder — documented MVP subset
(task T2.2).

Decodes little-endian 32-bit instruction words into a `DecodedInsn`.
Instructions outside the supported subset decode with mnemonic
"unsupported" so callers can report them precisely by address.
"""
from __future__ import annotations

from dataclasses import dataclass

REG_NAMES = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra",
]

_SPECIAL = {
    0x00: "sll", 0x02: "srl", 0x03: "sra", 0x04: "sllv", 0x06: "srlv", 0x07: "srav",
    0x08: "jr", 0x09: "jalr", 0x0A: "movz", 0x0B: "movn",
    0x0C: "syscall", 0x0D: "break", 0x0F: "sync",
    0x10: "mfhi", 0x11: "mthi", 0x12: "mflo", 0x13: "mtlo",
    0x18: "mult", 0x19: "multu", 0x1A: "div", 0x1B: "divu",
    0x20: "add", 0x21: "addu", 0x22: "sub", 0x23: "subu",
    0x24: "and", 0x25: "or", 0x26: "xor", 0x27: "nor",
    0x2A: "slt", 0x2B: "sltu",
}

_REGIMM = {
    0x00: "bltz", 0x01: "bgez", 0x02: "bltzl", 0x03: "bgezl",
    0x10: "bltzal", 0x11: "bgezal", 0x12: "bltzall", 0x13: "bgezall",
}

_OPCODE = {
    0x01: "regimm",
    0x02: "j", 0x03: "jal", 0x04: "beq", 0x05: "bne", 0x06: "blez", 0x07: "bgtz",
    0x08: "addi", 0x09: "addiu", 0x0A: "slti", 0x0B: "sltiu",
    0x0C: "andi", 0x0D: "ori", 0x0E: "xori", 0x0F: "lui",
    0x14: "beql", 0x15: "bnel", 0x16: "blezl", 0x17: "bgtzl",
    0x20: "lb", 0x21: "lh", 0x22: "lwl", 0x23: "lw", 0x24: "lbu", 0x25: "lhu",
    0x26: "lwr", 0x28: "sb", 0x29: "sh", 0x2A: "swl", 0x2B: "sw", 0x2E: "swr",
    0x2F: "cache", 0x33: "pref",
}

_BRANCHES = {
    "beq", "bne", "blez", "bgtz", "bltz", "bgez",
    "beql", "bnel", "blezl", "bgtzl",
    "bltzl", "bgezl", "bltzal", "bgezal", "bltzall", "bgezall",
}
_LIKELY = {"beql", "bnel", "blezl", "bgtzl", "bltzl", "bgezl", "bltzall", "bgezall"}
_BRANCH_LINKS = {"bltzal", "bgezal", "bltzall", "bgezall"}
_LOADS = {"lb", "lh", "lwl", "lw", "lbu", "lhu", "lwr"}
_STORES = {"sb", "sh", "swl", "sw", "swr"}
_JUMPS = {"j", "jal"}
_LINKERS = {"jal", "jalr"}


def sign_extend16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


@dataclass
class DecodedInsn:
    addr: int
    raw: int
    opcode: int
    mnemonic: str
    rs: int = 0
    rt: int = 0
    rd: int = 0
    shamt: int = 0
    imm: int = 0  # sign-extended 16-bit immediate (where applicable)
    uimm: int = 0  # zero-extended 16-bit immediate
    target: int = 0  # absolute jump target (j/jal)

    @property
    def is_branch(self) -> bool:
        return self.mnemonic in _BRANCHES

    @property
    def is_jump(self) -> bool:
        return self.mnemonic in _JUMPS

    @property
    def is_indirect_jump(self) -> bool:
        return self.mnemonic in ("jr", "jalr")

    @property
    def is_link(self) -> bool:
        return self.mnemonic in _LINKERS

    @property
    def is_likely(self) -> bool:
        """Branch-likely: the delay slot is nullified when not taken."""
        return self.mnemonic in _LIKELY

    @property
    def is_branch_link(self) -> bool:
        """REGIMM branch-and-link (writes $ra when the branch is taken)."""
        return self.mnemonic in _BRANCH_LINKS

    @property
    def has_delay_slot(self) -> bool:
        return self.is_branch or self.is_jump or self.is_indirect_jump

    @property
    def is_load(self) -> bool:
        return self.mnemonic in _LOADS

    @property
    def is_store(self) -> bool:
        return self.mnemonic in _STORES

    def disasm(self) -> str:
        m = self.mnemonic
        r = lambda n: f"${REG_NAMES[n]}"
        if m in ("sll", "srl", "sra"):
            return f"{m} {r(self.rd)}, {r(self.rt)}, {self.shamt}"
        if m in ("sllv", "srlv", "srav"):
            return f"{m} {r(self.rd)}, {r(self.rt)}, {r(self.rs)}"
        if m in ("jr", "jalr"):
            return f"{m} {r(self.rs)}" + (f", {r(self.rd)}" if m == "jalr" and self.rd != 31 else "")
        if m in ("mfhi", "mflo"):
            return f"{m} {r(self.rd)}"
        if m in ("mthi", "mtlo"):
            return f"{m} {r(self.rs)}"
        if m in ("mult", "multu", "div", "divu"):
            return f"{m} {r(self.rs)}, {r(self.rt)}"
        if m in ("syscall", "break", "sync"):
            return m
        if m in ("addu", "add", "subu", "sub", "and", "or", "xor", "nor", "slt", "sltu"):
            return f"{m} {r(self.rd)}, {r(self.rs)}, {r(self.rt)}"
        if m in ("movz", "movn"):
            return f"{m} {r(self.rd)}, {r(self.rs)}, {r(self.rt)}"
        if m in ("addiu", "addi", "slti", "sltiu", "andi", "ori", "xori"):
            if m in ("andi", "ori", "xori"):
                immstr = f"{self.uimm:#x}"
            else:
                immstr = f"-{(-self.imm):#x}" if self.imm < 0 else f"{self.imm:#x}"
            return f"{m} {r(self.rt)}, {r(self.rs)}, {immstr}"
        if m == "lui":
            return f"lui {r(self.rt)}, {hex(self.uimm)}"
        if m in _LOADS or m in _STORES:
            return f"{m} {r(self.rt)}, {self.imm}({r(self.rs)})"
        if m in ("cache", "pref"):
            return f"{m} {self.rt}, {self.imm}({r(self.rs)})"
        if m in ("beq", "bne", "beql", "bnel"):
            return f"{m} {r(self.rs)}, {r(self.rt)}, 0x{self.addr + 4 + self.imm * 4:08X}"
        if m in ("blez", "bgtz", "blezl", "bgtzl", "bltz", "bgez", "bltzl", "bgezl",
                 "bltzal", "bgezal", "bltzall", "bgezall"):
            return f"{m} {r(self.rs)}, 0x{self.addr + 4 + self.imm * 4:08X}"
        if m in _JUMPS:
            return f"{m} 0x{self.target:08X}"
        return f".word 0x{self.raw:08X}"


def decode(word: int, addr: int) -> DecodedInsn:
    """Decode one little-endian R5900 instruction word."""
    word &= 0xFFFFFFFF
    opcode = (word >> 26) & 0x3F
    rs = (word >> 21) & 0x1F
    rt = (word >> 16) & 0x1F
    rd = (word >> 11) & 0x1F
    shamt = (word >> 6) & 0x1F
    funct = word & 0x3F
    imm = sign_extend16(word)
    uimm = word & 0xFFFF

    if word == 0:
        return DecodedInsn(addr, word, 0, "nop")

    if opcode == 0x00:  # SPECIAL
        mnemonic = _SPECIAL.get(funct, "unsupported")
        return DecodedInsn(addr, word, opcode, mnemonic, rs, rt, rd, shamt, imm, uimm)

    if opcode == 0x01:  # REGIMM
        rt_field = rt
        mnemonic = _REGIMM.get(rt_field, "unsupported")
        return DecodedInsn(addr, word, opcode, mnemonic, rs, rt, rd, shamt, imm, uimm)

    if opcode in (0x02, 0x03):  # j / jal
        target = ((addr + 4) & 0xF0000000) | (((word & 0x03FFFFFF) << 2) & 0xFFFFFFFF)
        mnemonic = _OPCODE[opcode]
        return DecodedInsn(addr, word, opcode, mnemonic, rs, rt, rd, shamt, imm, uimm, target)

    mnemonic = _OPCODE.get(opcode, "unsupported")
    return DecodedInsn(addr, word, opcode, mnemonic, rs, rt, rd, shamt, imm, uimm)
