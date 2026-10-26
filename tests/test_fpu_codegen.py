"""Differential tests for native COP1/FPU lowering (T3.8)."""
import random
import struct

from tests import asm as a
from tests.difftest import make_state, run_case


def f32(raw: int) -> float:
    return struct.unpack("<f", (raw & 0xFFFFFFFF).to_bytes(4, "little"))[0]


def bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", value), "little")


def cop1(fmt, funct, fd, fs, ft):
    return a.enc_r(0x11, fmt, ft, fs, fd, funct)


def _state(f0=0.0, f1=0.0):
    st = make_state()
    st.fpr[0] = bits(f0)
    st.fpr[1] = bits(f1)
    return st


ARITH = [("add.s", 0x00), ("sub.s", 0x01), ("mul.s", 0x02),
         ("div.s", 0x03)]


def test_fpu_arithmetic_matches_golden():
    for name, funct in ARITH:
        for seed in range(20):
            rnd = random.Random(seed)
            x = rnd.randint(-512, 512) / 8.0
            y = float(rnd.choice((1, 2, 4, 8, 16))) * rnd.choice((-1, 1))
            st = _state(x, y)
            run_case([
                cop1(0x10, funct, 2, 0, 1),
                cop1(0x10, 0x06, 3, 0, 2),   # mov.s f3, f0
                a.jr(a.RA),
                a.NOP,
            ], state=st, label=f"{name} seed={seed}")


def test_fpu_bitwise_and_conversions():
    for seed in range(10):
        rnd = random.Random(seed)
        x = rnd.randint(-1000, 1000) / 4.0
        st = _state(x, 1.0)
        st.fpr[1] = rnd.getrandbits(32)
        run_case([
            cop1(0x10, 0x05, 2, 0, 0),   # abs.s
            cop1(0x10, 0x07, 3, 0, 0),   # neg.s
            cop1(0x10, 0x24, 4, 0, 0),   # cvt.w.s
            cop1(0x10, 0x24, 5, 1, 0),   # cvt.w.s (raw bits)
            cop1(0x14, 0x20, 6, 4, 0),   # cvt.s.w
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"conv seed={seed}")


def test_fpu_cvt_w_s_saturates():
    cases = [(3.9, 3), (-3.9, 0xFFFFFFFD), (1e30, 0x7FFFFFFF),
             (-1e30, 0x80000000)]
    for value, expected in cases:
        st = _state(value, 0.0)
        st.fpr[3] = 0
        run_case([
            cop1(0x10, 0x24, 3, 0, 0),
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"cvt {value}")
        # the differential harness already checks the native result; assert
        # the golden interpretation explicitly for documentation
        from ps2toolkit.golden.interpreter import step
        from ps2toolkit.golden.interpreter import EEMemory
        from ps2toolkit.r5900.decoder import decode
        mem = EEMemory()
        a.assemble(mem, 0x100000, cop1(0x10, 0x24, 3, 0, 0), a.NOP)
        step(st, mem)
        assert st.fpr[3] == expected


def test_fpu_sqrt_and_rsqrt():
    for seed in range(8):
        rnd = random.Random(seed)
        y = float(rnd.choice((1, 4, 9, 16, 25)))
        st = _state(2.0, y)
        run_case([
            cop1(0x10, 0x04, 2, 0, 1),   # sqrt.s f2 = sqrt(f1)
            cop1(0x10, 0x16, 3, 0, 1),   # rsqrt.s f3 = f0/sqrt(f1)
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"sqrt seed={seed}")


def test_fpu_accumulator_ops():
    for seed in range(10):
        rnd = random.Random(seed)
        st = _state(rnd.randint(-64, 64) / 4.0, rnd.randint(1, 16) / 4.0)
        st.fpu_acc = bits(rnd.randint(-64, 64) / 4.0)
        run_case([
            cop1(0x10, 0x1A, 0, 0, 1),   # mula.s: ACC = f0*f1
            cop1(0x10, 0x1C, 2, 0, 1),   # madd.s: f2 = ACC + f0*f1
            cop1(0x10, 0x18, 0, 0, 1),   # adda.s: ACC = f0+f1
            cop1(0x10, 0x1D, 3, 0, 1),   # msub.s: f3 = ACC - f0*f1
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"acc seed={seed}")


def test_mfc1_mtc1_cfc1_ctc1():
    st = make_state()
    st.fpr[0] = 0xFFFFFFFF
    st.regs[a.T0] = 0x12345678
    st.regs[a.T1] = 0x00800000
    run_case([
        a.enc_r(0x11, 0, a.T2, 0, 0, 0),    # mfc1 $t2, $f0
        a.enc_r(0x11, 4, a.T0, 1, 0, 0),    # mtc1 $t0, $f1
        a.enc_r(0x11, 6, a.T1, 31, 0, 0),   # ctc1 $t1, $31
        a.enc_r(0x11, 2, a.T3, 31, 0, 0),   # cfc1 $t3, $31
        a.jr(a.RA),
        a.NOP,
    ], state=st, label="cop1 moves")


def test_lwc1_swc1():
    st = make_state()
    st.fpr[0] = 0xDEADBEEF
    st.regs[a.T0] = 0x00110000
    run_case([
        a.enc_i(0x39, a.T0, 0, 0),   # swc1 f0, 0(t0)
        a.enc_i(0x31, a.T0, 1, 4),   # lwc1 f1, 4(t0)
        a.jr(a.RA),
        a.NOP,
    ], state=st, label="lwc1/swc1")


def test_compare_and_bc1_branches():
    # c.lt.s true -> bc1t taken, delay executes
    st = _state(1.0, 2.0)
    run_case([
        cop1(0x10, 0x34, 0, 0, 1),
        a.enc_i(0x11, 8, 1, 2),      # bc1t -> +16
        a.addiu(a.T0, a.ZERO, 1),    # delay
        a.addiu(a.T1, a.ZERO, 99),   # skipped
        a.addiu(a.T2, a.ZERO, 2),    # target
        a.jr(a.RA),
        a.NOP,
    ], state=st, label="bc1t taken")

    # c.lt.s false -> bc1f taken
    st = _state(2.0, 1.0)
    run_case([
        cop1(0x10, 0x34, 0, 0, 1),
        a.enc_i(0x11, 8, 0, 2),      # bc1f -> +16
        a.addiu(a.T0, a.ZERO, 1),
        a.addiu(a.T1, a.ZERO, 99),
        a.addiu(a.T2, a.ZERO, 2),
        a.jr(a.RA),
        a.NOP,
    ], state=st, label="bc1f taken")

    # C = 0 -> likely bc1tl not taken: delay slot nullified
    st = _state(1.0, 2.0)  # c.lt true -> wait, use c.lt false for C=0
    st = _state(2.0, 1.0)
    run_case([
        cop1(0x10, 0x34, 0, 0, 1),
        a.enc_i(0x11, 8, 3, 2),      # bc1tl -> +16, not taken
        a.addiu(a.T0, a.ZERO, 1),    # delay: nullified
        a.addiu(a.T1, a.ZERO, 2),
        a.addiu(a.T2, a.ZERO, 3),
        a.jr(a.RA),
        a.NOP,
    ], state=st, label="bc1tl not taken")


def test_fpu_min_max():
    for seed in range(10):
        rnd = random.Random(seed)
        x = rnd.randint(-100, 100) / 4.0
        y = rnd.randint(-100, 100) / 4.0
        st = _state(x, y)
        run_case([
            cop1(0x10, 0x28, 2, 0, 1),   # max.s
            cop1(0x10, 0x29, 3, 0, 1),   # min.s
            a.jr(a.RA),
            a.NOP,
        ], state=st, label=f"minmax seed={seed}")
