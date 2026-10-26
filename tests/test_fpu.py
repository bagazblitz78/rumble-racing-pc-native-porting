"""R5900 COP1 (FPU) semantics (T3.6), cross-checked against PCSX2."""
import struct

from ps2toolkit.golden.interpreter import EEMemory, EEState, step
from tests import asm as a

CODE = 0x00100000
STACK = 0x00110000


def f32(raw: int) -> float:
    return struct.unpack("<f", (raw & 0xFFFFFFFF).to_bytes(4, "little"))[0]


def bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", value), "little")


def cop1(fmt: int, funct: int, fd: int, fs: int, ft: int) -> int:
    """COP1 arithmetic: fs/rd, ft/rt, fd/sa."""
    return a.enc_r(0x11, fmt, ft, fs, fd, funct)


def make():
    return EEState(pc=CODE), EEMemory()


def exec1(word: int, **regs):
    st, mem = make()
    for name, value in regs.items():
        st.regs[getattr(a, name.upper())] = value
    a.assemble(mem, CODE, word, a.NOP)
    step(st, mem)
    return st


def test_fpu_arith_and_moves():
    st, mem = make()
    st.fpr[0] = bits(1.5)
    st.fpr[1] = bits(2.25)
    a.assemble(
        mem, CODE,
        cop1(0x10, 0x00, 2, 0, 1),   # add.s  f2 = f0 + f1
        cop1(0x10, 0x01, 3, 0, 1),   # sub.s  f3 = f0 - f1
        cop1(0x10, 0x02, 4, 0, 1),   # mul.s  f4 = f0 * f1
        cop1(0x10, 0x03, 5, 0, 1),   # div.s  f5 = f0 / f1
        cop1(0x10, 0x06, 6, 0, 0),   # mov.s  f6 = f0
        cop1(0x10, 0x07, 7, 0, 0),   # neg.s  f7 = -f0
        cop1(0x10, 0x05, 8, 0, 0),   # abs.s  f8 = |f0|
        cop1(0x10, 0x04, 9, 0, 1),   # sqrt.s f9 = sqrt(f1)
        a.NOP,
    )
    for _ in range(8):
        step(st, mem)
    assert f32(st.fpr[2]) == 3.75
    assert f32(st.fpr[3]) == -0.75
    assert f32(st.fpr[4]) == 3.375
    assert abs(f32(st.fpr[5]) - 1.5 / 2.25) < 1e-6
    assert st.fpr[6] == st.fpr[0]
    assert f32(st.fpr[7]) == -1.5
    assert f32(st.fpr[8]) == 1.5
    assert abs(f32(st.fpr[9]) - 1.5) < 1e-6


def test_division_by_zero_clamps_like_pcsx2():
    st, mem = make()
    st.fpr[0] = bits(1.0)
    st.fpr[1] = 0
    a.assemble(mem, CODE, cop1(0x10, 0x03, 2, 0, 1), a.NOP)
    step(st, mem)
    assert st.fpr[2] == 0x7F7FFFFF  # +FLT_MAX, not infinity
    assert st.fcr31 & 0x00010000   # FPUflagD


def test_cvt_w_s_truncates_and_saturates():
    st, mem = make()
    st.fpr[0] = bits(3.9)
    st.fpr[1] = bits(-3.9)
    st.fpr[2] = bits(1e30)
    st.fpr[3] = bits(-1e30)
    a.assemble(
        mem, CODE,
        cop1(0x10, 0x24, 4, 0, 0),   # cvt.w.s f4, f0
        cop1(0x10, 0x24, 5, 1, 0),   # cvt.w.s f5, f1
        cop1(0x10, 0x24, 6, 2, 0),
        cop1(0x10, 0x24, 7, 3, 0),
        a.NOP,
    )
    for _ in range(4):
        step(st, mem)
    assert st.fpr[4] == 3
    assert st.fpr[5] == 0xFFFFFFFD  # -3 (truncation toward zero)
    assert st.fpr[6] == 0x7FFFFFFF
    assert st.fpr[7] == 0x80000000


def test_cvt_s_w():
    st, mem = make()
    st.fpr[0] = 0xFFFFFFFE  # -2 as a word
    st.fpr[1] = 7
    a.assemble(mem, CODE,
               cop1(0x14, 0x20, 2, 0, 0),  # cvt.s.w f2, f0
               cop1(0x14, 0x20, 3, 1, 0),  # cvt.s.w f3, f1
               a.NOP)
    step(st, mem)
    step(st, mem)
    assert f32(st.fpr[2]) == -2.0
    assert f32(st.fpr[3]) == 7.0


def test_mfc1_mtc1_cfc1_ctc1():
    st, mem = make()
    st.fpr[0] = 0xFFFFFFFF
    st.regs[a.T0] = 0x12345678
    st.regs[a.T1] = 0x00800000
    a.assemble(
        mem, CODE,
        a.enc_r(0x11, 0, a.T2, 0, 0, 0),    # mfc1 $t2, $f0
        a.enc_r(0x11, 4, a.T0, 1, 0, 0),    # mtc1 $t0, $f1
        a.enc_r(0x11, 6, a.T1, 31, 0, 0),   # ctc1 $t1, $31
        a.enc_r(0x11, 2, a.T3, 31, 0, 0),   # cfc1 $t3, $31
        a.NOP,
    )
    step(st, mem)
    assert st.regs[a.T2] & 0xFFFFFFFFFFFFFFFF == 0xFFFFFFFFFFFFFFFF  # sign-extended
    step(st, mem)
    assert st.fpr[1] == 0x12345678
    step(st, mem)
    assert st.fcr31 == 0x00800000
    step(st, mem)
    assert st.regs[a.T3] == 0x00800000


def test_compare_sets_condition_and_bc1_branches():
    st, mem = make()
    st.fpr[0] = bits(1.0)
    st.fpr[1] = bits(2.0)
    a.assemble(
        mem, CODE,
        cop1(0x10, 0x34, 0, 0, 1),   # c.lt.s f0, f1 -> true
        a.enc_r(0x11, 8, 1, 0, 0, 0),  # bc1t +? (offset filled below)
        a.addiu(a.T0, a.ZERO, 1),    # delay
        a.addiu(a.T1, a.ZERO, 99),   # skipped
        a.NOP,
    )
    # patch the branch offset: bc1t at CODE+4, target CODE+16 -> imm 2
    mem.write(CODE + 4, (0x45010002).to_bytes(4, "little"))
    step(st, mem)          # c.lt.s
    assert st.fcr31 & 0x00800000
    step(st, mem)          # bc1t + delay
    assert st.pc == CODE + 16
    assert st.regs[a.T0] == 1
    assert st.regs[a.T1] == 0


def test_bc1f_not_taken_and_likely_nullifies_delay():
    st, mem = make()
    st.fcr31 = 0
    a.assemble(
        mem, CODE,
        a.enc_r(0x11, 8, 3, 0, 0, 0),  # bc1tl with C=0 -> not taken
        a.addiu(a.T0, a.ZERO, 1),      # delay: nullified
        a.addiu(a.T1, a.ZERO, 2),      # fallthrough
        a.NOP,
    )
    step(st, mem)
    assert st.pc == CODE + 8
    assert st.regs[a.T0] == 0
    step(st, mem)
    assert st.regs[a.T1] == 2


def test_acc_and_madd():
    st, mem = make()
    st.fpr[0] = bits(2.0)
    st.fpr[1] = bits(3.0)
    st.fpu_acc = bits(10.0)
    a.assemble(
        mem, CODE,
        cop1(0x10, 0x1C, 2, 0, 1),   # madd.s f2 = ACC + f0*f1 = 16
        cop1(0x10, 0x16, 3, 0, 1),   # rsqrt.s f3 = f0 / sqrt(f1) (approx)
        a.NOP,
    )
    step(st, mem)
    assert f32(st.fpr[2]) == 16.0
    step(st, mem)
    assert abs(f32(st.fpr[3]) - 2.0 / (3.0 ** 0.5)) < 1e-6


def test_lwc1_swc1_little_endian():
    st, mem = make()
    st.regs[a.T0] = STACK
    st.fpr[0] = 0xDEADBEEF
    a.assemble(
        mem, CODE,
        a.enc_i(0x39, a.T0, 0, 0),   # swc1 f0, 0(t0)
        a.enc_i(0x31, a.T0, 1, 4),   # lwc1 f1, 4(t0)
        a.NOP,
    )
    mem.write(STACK + 4, 0x3F800000.to_bytes(4, "little"))
    step(st, mem)
    assert mem.read(STACK, 4) == b"\xEF\xBE\xAD\xDE"
    step(st, mem)
    assert st.fpr[1] == 0x3F800000  # 1.0f
