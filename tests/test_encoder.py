"""Tests for the x64 machine-code encoder (T2.5).

Emitted bytes are verified with capstone — an independent disassembler —
so tests check real x86-64 semantics, not our own bookkeeping.
"""
import pytest

from ps2toolkit.x64emit.encoder import (
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
    Asm, Label,
)

md = None


def disasm(code: bytes) -> list[str]:
    global md
    if md is None:
        import capstone
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = False
    return [f"{i.mnemonic} {i.op_str}".strip() for i in md.disasm(code, 0x1000)]


def test_mov_ri_zero_extends():
    a = Asm()
    a.mov_ri(RCX, 0xDEADBEEF)
    insns = disasm(bytes(a.buf))
    assert insns == ["mov ecx, 0xdeadbeef"]


def test_mov_ri_high_register():
    a = Asm()
    a.mov_ri(R10, 0x1234)
    assert disasm(bytes(a.buf)) == ["mov r10d, 0x1234"]


def test_mov_rr():
    a = Asm()
    a.mov_rr(R15, RCX)
    assert disasm(bytes(a.buf)) == ["mov r15, rcx"]


def test_ld_slot_and_st_slot():
    a = Asm()
    a.ld32(RAX, R15, 72)   # guest reg 9 slot
    a.st32(R15, 128, RCX)  # guest reg 16 slot
    insns = disasm(bytes(a.buf))
    assert insns[0] == "mov eax, dword ptr [r15 + 0x48]"
    assert insns[1] == "mov dword ptr [r15 + 0x80], ecx"


def test_ld_ram_st_ram():
    a = Asm()
    a.ld32_ram(RDX, R14, RCX)
    a.st32_ram(R14, RCX, R8)
    insns = disasm(bytes(a.buf))
    assert insns[0] == "mov edx, dword ptr [r14 + rcx]"
    assert insns[1] == "mov dword ptr [r14 + rcx], r8d"


def test_alu_forms():
    a = Asm()
    a.alu_rr("add", RAX, RCX)
    a.alu_rr("sub", R8, R9)
    a.alu_ri("cmp", RDX, -4)
    a.alu_ri("and", R11, 0xFF)
    insns = disasm(bytes(a.buf))
    assert insns == ["add eax, ecx", "sub r8d, r9d", "cmp edx, 0xfffffffc", "and r11d, 0xff"]


def test_shift_forms():
    a = Asm()
    a.shift_ri("shl", RCX, 4)
    a.shift_ri("sar", RDX, 31)
    a.shift_cl("shr", R8)
    insns = disasm(bytes(a.buf))
    assert insns == ["shl ecx, 4", "sar edx, 0x1f", "shr r8d, cl"]


def test_setcc_and_test():
    a = Asm()
    a.setcc("e", R10)
    a.test_rr(R10, R10)
    a.movzx(RAX, R10)
    insns = disasm(bytes(a.buf))
    assert insns[0] == "sete r10b"
    assert insns[1] == "test r10d, r10d"
    assert insns[2] == "movzx eax, r10b"


def test_bswap_and_mul_family():
    a = Asm()
    a.bswap(RCX)
    a.imul_rr(RAX, RCX)
    a.mul_r(RCX)
    a.cdq()
    a.idiv_r(RCX)
    a.neg_r(RDX)
    a.movsxd(RAX, RCX)
    insns = disasm(bytes(a.buf))
    assert insns == [
        "bswap ecx",
        "imul eax, ecx",
        "mul ecx",
        "cdq",
        "idiv ecx",
        "neg edx",
        "movsxd rax, ecx",
    ]


def test_push_pop():
    a = Asm()
    a.push(R14)
    a.push(R15)
    a.pop(R15)
    a.pop(R14)
    a.ret()
    assert disasm(bytes(a.buf)) == ["push r14", "push r15", "pop r15", "pop r14", "ret"]


def test_labels_and_rel32_fixups():
    a = Asm()
    done = a.new_label()
    a.jmp(done)            # forward jump, 5-byte placeholder
    a.nop()                # offset 5
    a.bind(done)           # offset 6
    a.jcc("e", done)       # backward jump to bound label
    a.ret()
    code = bytes(a.buf)
    assert code[0] == 0xE9
    insns = disasm(code)
    # jmp: rip=0x1005, rel=+1 -> 0x1006; je: rip=0x1010... bound label at 0x1006
    assert insns[0] == "jmp 0x1006"
    assert insns[1] == "nop"
    assert insns[2] == "je 0x1006"
    assert insns[3] == "ret"


def test_unbound_label_patched_later():
    a = Asm()
    target = a.new_label()
    a.jcc("ne", target)    # unbound: placeholder
    a.ret()
    a.nop()
    a.bind(target)
    a.ret()
    insns = disasm(bytes(a.buf))
    # je at 0x1000 (6 bytes) -> rip 0x1006; ret@6, nop@7 -> target at offset 8
    assert insns[0] == "jne 0x1008"
    assert insns[-1] == "ret"
