"""Control-flow graph construction with delay-slot folding (task T2.4).

Builds per-function CFGs from a decoded instruction stream. Branch and
jump delay-slot instructions are folded into the block terminator: they
execute on BOTH paths, exactly like the golden interpreter does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..golden.interpreter import UnsupportedInstruction
from ..r5900.decoder import DecodedInsn

InsnAt = Callable[[int], DecodedInsn]
IsFunctionStart = Callable[[int], bool]


@dataclass
class Block:
    addr: int
    insns: list[tuple[int, DecodedInsn]] = field(default_factory=list)
    term: "Term | None" = None


@dataclass
class Term:
    pass


@dataclass
class BranchTerm(Term):
    insn: DecodedInsn
    target: int
    fallthrough: int
    delay: DecodedInsn


@dataclass
class JumpTerm(Term):
    insn: DecodedInsn
    target: int
    delay: DecodedInsn


@dataclass
class CallTerm(Term):
    insn: DecodedInsn
    target: int
    fallthrough: int
    delay: DecodedInsn


@dataclass
class ReturnTerm(Term):
    insn: DecodedInsn  # jr $ra
    delay: DecodedInsn


@dataclass
class IndirectTerm(Term):
    insn: DecodedInsn  # jr with non-ra register
    delay: DecodedInsn


@dataclass
class StopTerm(Term):
    """Execution reaches an explicit stop address."""


@dataclass
class Function:
    entry: int
    blocks: dict[int, Block] = field(default_factory=dict)


def _raise_unsupported(insn: DecodedInsn):
    raise UnsupportedInstruction(insn.addr, insn.raw,
                                 insn.mnemonic if insn.mnemonic != "unsupported"
                                 else f"op={insn.opcode:#x}")


def build_function(insn_at: InsnAt, entry: int, is_function_start: IsFunctionStart,
                   stop_addrs: set[int]) -> Function:
    """Build the CFG of one function.

    - `is_function_start(addr)` marks entries of other functions (jal
      targets): control never falls through into them.
    - `stop_addrs` are addresses where execution halts (StopTerm).
    Raises UnsupportedInstruction / ValueError for untranslatable input.
    """
    fn = Function(entry=entry)
    starts: set[int] = {entry}
    pending: list[int] = [entry]
    while pending:
        start = pending.pop(0)
        if start in fn.blocks:
            continue
        block = Block(addr=start)
        fn.blocks[start] = block
        addr = start
        while True:
            if addr in stop_addrs:
                block.term = StopTerm()
                break
            if addr != start and addr in starts:
                # reached a known block start: fall through into it
                block.term = FallTo(addr)
                break
            insn = insn_at(addr)
            if insn.mnemonic == "unsupported":
                _raise_unsupported(insn)

            if insn.is_branch:
                delay = insn_at(addr + 4)
                if delay.mnemonic == "unsupported":
                    _raise_unsupported(delay)
                if delay.has_delay_slot:
                    raise ValueError(f"branch in delay slot at {delay.addr:#x}")
                target = addr + 4 + insn.imm * 4
                if target % 4:
                    raise ValueError(f"misaligned branch target {target:#x}")
                if stop_addrs and target > max(stop_addrs):
                    raise ValueError(
                        f"branch target {target:#x} beyond region "
                        f"(stop {max(stop_addrs):#x}) at {addr:#x}")
                block.term = BranchTerm(insn, target, addr + 8, delay)
                for t in (target, addr + 8):
                    if not is_function_start(t) and t not in starts:
                        starts.add(t)
                        pending.append(t)
                break

            if insn.is_jump:  # j / jal
                delay = insn_at(addr + 4)
                if delay.mnemonic == "unsupported":
                    _raise_unsupported(delay)
                if delay.has_delay_slot:
                    raise ValueError(f"branch in delay slot at {delay.addr:#x}")
                if insn.is_link:
                    block.term = CallTerm(insn, insn.target, addr + 8, delay)
                    if not is_function_start(insn.target) and insn.target not in starts:
                        starts.add(insn.target)
                        pending.append(insn.target)
                else:
                    block.term = JumpTerm(insn, insn.target, delay)
                    if insn.target not in starts:
                        starts.add(insn.target)
                        pending.append(insn.target)
                # control returns to the instruction after the delay slot
                if not is_function_start(addr + 8) and (addr + 8) not in starts:
                    starts.add(addr + 8)
                    pending.append(addr + 8)
                break

            if insn.is_indirect_jump:
                delay = insn_at(addr + 4)
                if delay.mnemonic == "unsupported":
                    _raise_unsupported(delay)
                if delay.has_delay_slot:
                    raise ValueError(f"branch in delay slot at {delay.addr:#x}")
                if insn.rs == 31:
                    block.term = ReturnTerm(insn, delay)
                else:
                    block.term = IndirectTerm(insn, delay)
                break

            block.insns.append((addr, insn))
            addr += 4

    return fn


@dataclass
class FallTo(Term):
    target: int
