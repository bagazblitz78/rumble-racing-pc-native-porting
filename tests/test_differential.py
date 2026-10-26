"""Randomized differential testing (milestone M-B).

Generates >=1000 random EE programs (ALU, shifts, mult/div incl. PS2 quirks,
big-endian memory incl. unaligned LWL/LWR/SWL/SWR, forward branches with
delay slots), compiles each to native x64, executes it natively, and
requires bit-for-bit equality with the golden interpreter on every piece of
architectural state (32 GPRs, HI, LO) plus the exit code.

Branches are forward-only so every program terminates; every memory access
stays inside the stack window (SP-relative, offsets 0..252), so no program
can fault either side differently.
"""
import random

import pytest

from tests import asm as a
from tests.difftest import CLOBBERABLE, run_case

READABLE = list(range(29))  # 0..28; $zero reads as 0 on both sides
SHIFTS = ("sll", "srl", "sra")


def _simple(rnd: random.Random) -> int:
    """One non-branch, non-delay-carrying instruction."""
    kind = rnd.randrange(18)
    rd = rnd.choice(CLOBBERABLE)
    rs, rt = rnd.choice(READABLE), rnd.choice(READABLE)
    if kind == 0:
        return a.addu(rd, rs, rt)
    if kind == 1:
        return a.subu(rd, rs, rt)
    if kind == 2:
        return a.and_(rd, rs, rt)
    if kind == 3:
        return a.or_(rd, rs, rt)
    if kind == 4:
        return a.xor_(rd, rs, rt)
    if kind == 5:
        return a.nor_(rd, rs, rt)
    if kind == 6:
        return a.slt(rd, rs, rt)
    if kind == 7:
        return a.sltu(rd, rs, rt)
    if kind == 8:
        return a.addiu(rnd.choice(CLOBBERABLE), rs, rnd.randint(-32768, 32767))
    if kind == 9:
        op = rnd.choice(("andi", "ori", "xori"))
        return getattr(a, op)(rnd.choice(CLOBBERABLE), rs, rnd.randint(0, 65535))
    if kind == 10:
        return a.lui(rnd.choice(CLOBBERABLE), rnd.randint(0, 65535))
    if kind == 11:
        return getattr(a, rnd.choice(SHIFTS))(rd, rt, rnd.randrange(32))
    if kind == 12:
        return rnd.choice((a.sllv, a.srlv, a.srav))(rd, rt, rs)
    if kind == 13:
        return rnd.choice((a.mult, a.multu, a.div, a.divu))(rs, rt)
    if kind == 14:
        return rnd.choice((a.mflo, a.mfhi))(rd)
    if kind == 15:
        return rnd.choice((a.movz, a.movn))(rd, rs, rt)
    if kind == 16:
        return rnd.choice((
            a.sync(),
            a.cache(rnd.randrange(32), rnd.randrange(0, 64), a.SP),
            a.pref(rnd.randrange(32), rnd.randrange(0, 64), a.SP),
        ))
    # memory
    return _memory(rnd, rd, rs)


def _memory(rnd: random.Random, rd: int, rs: int) -> int:
    kind = rnd.randrange(8)
    if kind == 0:
        return a.sw(rnd.choice(READABLE), rnd.randrange(64) * 4, a.SP)
    if kind == 1:
        return a.lw(rd, rnd.randrange(64) * 4, a.SP)
    if kind == 2:
        return a.sb(rnd.choice(READABLE), rnd.randrange(256), a.SP)
    if kind == 3:
        return a.lbu(rd, rnd.randrange(256), a.SP)
    if kind == 4:
        return a.sh(rnd.choice(READABLE), rnd.randrange(128) * 2, a.SP)
    if kind == 5:
        return a.lhu(rd, rnd.randrange(128) * 2, a.SP)
    if kind == 6:
        return rnd.choice((a.lwl, a.lwr))(rd, rnd.randrange(253), a.SP)
    return rnd.choice((a.swl, a.swr))(rnd.choice(READABLE),
                                       rnd.randrange(253), a.SP)


def _branch(rnd: random.Random) -> int:
    skip = rnd.randint(1, 3)
    if rnd.random() < 0.5:
        return a.beq(rnd.choice(READABLE), rnd.choice(READABLE), skip)
    op = rnd.choice((a.bne, a.blez, a.bgtz, a.bltz, a.bgez))
    if op is a.bne:
        return op(rnd.choice(READABLE), rnd.choice(READABLE), skip)
    return op(rnd.choice(READABLE), skip)


def gen_program(rnd: random.Random) -> list[int]:
    words: list[int] = []
    target_len = rnd.randint(10, 34)
    while len(words) < target_len:
        # only branch with headroom so no branch target can overshoot the
        # program end (the terminator appended below is 1-2 words)
        if rnd.random() < 0.30 and len(words) + 6 < target_len:
            words.append(_branch(rnd))
            words.append(_simple(rnd))  # delay slot: never a branch
        else:
            words.append(_simple(rnd))
    if rnd.random() < 0.12:  # terminate via syscall exit code instead
        code = rnd.randint(0, 0x3FF)
        words.append(a.enc_r(0, 0, 0, (code >> 5) & 0x1F, code & 0x1F, 12))
    else:
        words.append(a.jr(a.RA))
        words.append(a.NOP)
    return words


def test_random_programs_match_golden():
    failures = []
    for seed in range(1000):
        rnd = random.Random(seed)
        words = gen_program(rnd)
        try:
            run_case(words, seed=seed, label=f"seed={seed}")
        except AssertionError as exc:
            failures.append((seed, exc))
            if len(failures) >= 3:
                break
    assert not failures, (
        "randomized differential failures:\n"
        + "\n".join(f"seed {s}: {e}" for s, e in failures)
    )


def _branch_case(branch_word, t0=3, t1=3):
    """Both branch edges end at a syscall; the delay slot writes $t2."""
    return [
        a.addiu(a.T0, a.ZERO, t0),
        a.addiu(a.T1, a.ZERO, t1),
        branch_word,
        a.addiu(a.T2, a.ZERO, 0x11),   # delay slot
        a.addiu(a.T3, a.ZERO, 0x22),   # fallthrough
        a.addiu(a.T5, a.ZERO, 0x55),
        a.addiu(a.T4, a.ZERO, 0x33),   # branch target
        a.syscall(),
    ]


LIKELY_CASES = [
    ("beql_taken", 3, 3, a.beql(a.T0, a.T1, 3)),
    ("beql_not_taken", 3, 4, a.beql(a.T0, a.T1, 3)),
    ("bnel_taken", 3, 4, a.bnel(a.T0, a.T1, 3)),
    ("bnel_not_taken", 3, 3, a.bnel(a.T0, a.T1, 3)),
    ("blezl_taken", -1, 0, a.blezl(a.T0, 3)),
    ("blezl_not_taken", 3, 0, a.blezl(a.T0, 3)),
    ("bgtzl_taken", 3, 0, a.bgtzl(a.T0, 3)),
    ("bgtzl_not_taken", 0, 0, a.bgtzl(a.T0, 3)),
    ("bltzl_taken", -1, 0, a.bltzl(a.T0, 3)),
    ("bltzl_not_taken", 3, 0, a.bltzl(a.T0, 3)),
    ("bgezl_taken", 3, 0, a.bgezl(a.T0, 3)),
    ("bgezl_not_taken", -1, 0, a.bgezl(a.T0, 3)),
]

LINK_CASES = [
    ("bgezal_taken", 3, a.bgezal(a.T0, 3)),
    ("bgezal_not_taken", -1, a.bgezal(a.T0, 3)),
    ("bltzal_taken", -1, a.bltzal(a.T0, 3)),
    ("bltzal_not_taken", 3, a.bltzal(a.T0, 3)),
    ("bgezall_taken", 3, a.bgezall(a.T0, 3)),
    ("bgezall_not_taken", -1, a.bgezall(a.T0, 3)),
    ("bltzall_taken", -1, a.bltzall(a.T0, 3)),
    ("bltzall_not_taken", 3, a.bltzall(a.T0, 3)),
]


@pytest.mark.parametrize("name,t0,t1,branch", LIKELY_CASES,
                         ids=[c[0] for c in LIKELY_CASES])
def test_branch_likely_matches_golden(name, t0, t1, branch):
    run_case(_branch_case(branch, t0, t1), seed=7, label=name)


@pytest.mark.parametrize("name,t0,branch", LINK_CASES,
                         ids=[c[0] for c in LINK_CASES])
def test_branch_and_link_matches_golden(name, t0, branch):
    run_case(_branch_case(branch, t0, 0), seed=7, label=name)
