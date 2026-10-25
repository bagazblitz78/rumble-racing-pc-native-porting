"""Tests for CFG construction with delay-slot folding (T2.4)."""
import pytest

from ps2toolkit.golden.interpreter import EEMemory, EEState
from ps2toolkit.ir.cfg import (
    BranchTerm,
    CallTerm,
    IndirectTerm,
    ReturnTerm,
    StopTerm,
    build_function,
)

from tests import asm as a

CODE = 0x00100000


def build(*words, entry=CODE):
    mem = EEMemory()
    state = EEState(pc=entry)
    a.assemble(mem, entry, *words)
    return mem


def insn_at(mem, addr):
    from ps2toolkit.r5900.decoder import decode
    return decode(mem.read32(addr), addr)


def test_straight_line_block():
    mem = build(a.addiu(a.T0, a.ZERO, 1), a.addiu(a.T1, a.ZERO, 2), a.NOP)
    fn = build_function(lambda addr: insn_at(mem, addr), entry=CODE,
                        is_function_start=lambda addr: False,
                        stop_addrs={CODE + 12})
    blocks = fn.blocks
    assert len(blocks) == 1
    blk = blocks[CODE]
    assert [i.mnemonic for _, i in blk.insns] == ["addiu", "addiu", "nop"]
    assert isinstance(blk.term, StopTerm)


def test_branch_splits_blocks_and_folds_delay_slot():
    mem = build(
        a.beq(a.T0, a.T1, 2),       # CODE+0 -> target CODE+12
        a.addiu(a.T2, a.ZERO, 1),   # delay slot
        a.addiu(a.T3, a.ZERO, 2),   # CODE+8 fallthrough
        a.addiu(a.T4, a.ZERO, 3),   # CODE+12 target
        a.NOP,
    )
    fn = build_function(lambda addr: insn_at(mem, addr), entry=CODE,
                        is_function_start=lambda addr: False,
                        stop_addrs={CODE + 20})
    assert set(fn.blocks) == {CODE, CODE + 8, CODE + 12}
    term = fn.blocks[CODE].term
    assert isinstance(term, BranchTerm)
    assert term.target == CODE + 12
    assert term.fallthrough == CODE + 8
    assert term.delay.mnemonic == "addiu"  # folded into terminator
    assert term.delay.rt == a.T2
    # delay slot folded into terminator, not duplicated in successors
    assert [i.mnemonic for _, i in fn.blocks[CODE + 8].insns] == ["addiu"]
    assert [i.mnemonic for _, i in fn.blocks[CODE + 12].insns] == ["addiu", "nop"]


def test_jal_creates_call_term_and_callee_function():
    mem = build(
        a.jal(CODE + 0x40),
        a.NOP,                              # delay
        a.addiu(a.T9, a.ZERO, 1),           # CODE+8 fallthrough
        a.NOP,
        # filler to 0x40: 4 instructions so far + 12 nops = 16 words = 0x40
        *([a.NOP] * 12),
        a.addiu(a.T0, a.ZERO, 7),           # CODE+0x40 callee
        a.jr(a.RA),
        a.NOP,
        a.NOP,
    )
    starts = {CODE, CODE + 0x40}
    fn = build_function(lambda addr: insn_at(mem, addr), entry=CODE,
                        is_function_start=lambda addr: addr in starts,
                        stop_addrs={CODE + 8})
    callee = build_function(lambda addr: insn_at(mem, addr), entry=CODE + 0x40,
                            is_function_start=lambda addr: addr in starts,
                            stop_addrs=set())
    term = fn.blocks[CODE].term
    assert isinstance(term, CallTerm)
    assert term.target == CODE + 0x40
    assert term.delay.mnemonic == "nop"
    assert fn.blocks[CODE + 8].term is not None  # stop at CODE+8
    assert isinstance(fn.blocks[CODE + 8].term, StopTerm)
    # callee ends with return term (jr $ra)
    cterm = callee.blocks[CODE + 0x40].term
    assert isinstance(cterm, ReturnTerm)
    assert cterm.delay.mnemonic == "nop"


def test_jump_target_and_unsupported():
    mem = build(a.j(CODE + 0x10), a.NOP, a.NOP, a.NOP, a.NOP, a.NOP)
    fn = build_function(lambda addr: insn_at(mem, addr), entry=CODE,
                        is_function_start=lambda addr: False,
                        stop_addrs={CODE + 0x18})
    term = fn.blocks[CODE].term
    from ps2toolkit.ir.cfg import JumpTerm
    assert isinstance(term, JumpTerm)
    assert term.target == CODE + 0x10
    assert CODE + 0x10 in fn.blocks


def test_unsupported_instruction_reported():
    mem = build((28 << 26) | (8 << 21) | (9 << 16))  # MMI opcode -> unsupported
    from ps2toolkit.golden.interpreter import UnsupportedInstruction
    with pytest.raises(UnsupportedInstruction):
        build_function(lambda addr: insn_at(mem, addr), entry=CODE,
                       is_function_start=lambda addr: False, stop_addrs=set())


def test_indirect_jump_other_register():
    mem = build(a.jr(a.T0), a.NOP, a.NOP)
    fn = build_function(lambda addr: insn_at(mem, addr), entry=CODE,
                        is_function_start=lambda addr: False, stop_addrs=set())
    assert isinstance(fn.blocks[CODE].term, IndirectTerm)
