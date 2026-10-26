# Milestone 5 — Native COP1 / FPU lowering

Task T3.8 lowered the scalar FPU to SSE. Together with milestone 4 this
covers every scalar instruction the boot trace reaches: integer, MMI,
LQ/SQ and COP1 all execute as native x86-64.

## Context additions

| Field | Offset | Size |
|---|---|---|
| `fpr[32]` | 552 | 128 bytes (raw IEEE-754 single bits) |
| `fcr31` | 680 | 4 bytes |
| `fpu_acc` | 684 | 4 bytes |
| guest RAM | 688 (`0x2B0`) | 32 MiB |

## Lowered instructions

- Transfers: `MFC1`, `MTC1`, `CFC1`, `CTC1`, `LWC1`, `SWC1`
- Arithmetic: `ADD.S`, `SUB.S`, `MUL.S`, `DIV.S`, `SQRT.S`, `RSQRT.S`,
  `MAX.S`, `MIN.S`, `ABS.S`, `NEG.S`, `MOV.S`
- Accumulator: `ADDA.S`, `SUBA.S`, `MULA.S`, `MADDA.S`, `MSUBA.S`,
  `MADD.S`, `MSUB.S`
- Conversion: `CVT.S.W`, `CVT.W.S` (including the PS2's truncation and
  `0x7FFFFFFF`/`0x80000000` saturation outside the float range)
- Compare/branch: `C.F`, `C.EQ`, `C.LT`, `C.LE` set/clear FCR31 bit 23,
  and `BC1F/T/FL/TL` branch on it (likely forms nullify the delay slot)

FPU field roles follow the hardware: `fs = rd`, `ft = rt`, `fd = sa`.

## Evidence

- 9 differential test groups compare the full FPU register file, FCR31,
  ACC and GPRs between the golden interpreter and native execution:
  arithmetic, bitwise/conversions, CVT saturation, sqrt/rsqrt, accumulator
  chains, moves, LWC1/SWC1, compares + all four BC1 forms, min/max.
- A standalone PE test round-trips 42 through `MTC1 → CVT.S.W → CVT.W.S →
  MFC1` and exits with 42, proving COP1 runs natively inside the image.
- Full suite: 285 tests passing.

## Known limitation

Exception *cause* flags (I/D/O/U/SI/SD/SO/SU) are not modelled by the
native lowering; the condition bit and BC1 are exact. The golden model
tracks the cause flags. Normal-valued code matches exactly; code that
relies on FPU exception flags will need that pass next.

## Remaining gaps

- VU0/COP2 macro mode and VU memory (the only decoder gap).
- MMI `DIV1`, `PMADDW`/`PMULTW` accumulator forms and the remaining
  unsaturated/abs/word-min/max MMI operations.
- 44 unaligned 64-bit (`LDL/LDR/SDL/SDR`) words.
