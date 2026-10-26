# Milestone 4 — Native 128-bit GPRs, MMI and LQ/SQ

Task T3.7 rebuilt the recompiler context for the R5900's 128-bit general
registers and added the first SSE2 lowering of the multimedia instruction
set. The recompiler can now run the same MMI/LQ/SQ code the golden model
runs, and a standalone Windows PE built from it exits with a guest-computed
value.

## Context layout

| Field | Offset | Size |
|---|---|---|
| `regs[i]` | `i * 16` | 16 bytes (low u64 then high u64) |
| `hi` | 512 | 16 bytes |
| `lo` | 528 | 16 bytes |
| `ram_ptr` | 544 | 8 bytes |
| guest RAM | 560 (`0x230`) | 32 MiB (bss) |

Scalar results write the low 64 bits and sign-extend the upper 64, matching
the golden model's 128-bit container. MMI results and LQ/SQ write all 16
bytes. `logical AND/OR/XOR/NOR`, MOVZ/MOVN and the immediate logical forms
were re-lowered at 128-bit width so they match the golden model for
arbitrary upper lanes.

## Lowered MMI instructions

Packed arithmetic: `PADDB/H/W`, `PSUBB/H/W`, `PADDSB/H`, `PSUBSB/H`,
`PADDUB/H`, `PSUBUB/H` (SSE2 saturating forms).

Compare/min/max: `PCEQB/H/W`, `PCGTB/H/W`, `PMAXH`, `PMINH`.

Interleave/copy/shuffle: `PEXTLW`, `PEXTUW`, `PEXTLH`, `PEXTUH`, `PEXTLB`,
`PEXTUB`, `PCPYLD`, `PCPYUD`, `PCPYH`, `PEXEH`, `PREVH`, `PEXCH`, `PEXEW`,
`PROT3W`, `PEXCW`.

Bitwise: `PAND`, `POR`, `PXOR`, `PNOR`.

Scalar MMI helpers: `PLZCW`, `MULT1`, `MFHI1`, `MFLO1`, `MTHI1`, `MTLO1`.

`LQ`/`SQ` use `movdqu` between guest RAM and the register slots, so no
alignment requirement is introduced.

Not lowered yet: `DIV1`, `PMADDW`/`PMULTW` and the rest of the accumulator
family, `PABSW/H`, `PMAXW`/`PMINW`, the 32-bit saturating forms, and the
VU0/COP2 space. `compile_program` raises `CodeGenError` with the exact
mnemonic instead of emitting wrong code.

## Evidence

- 45 new differential tests compare full 128-bit GPRs, HI/LO and memory
  between the golden interpreter and native execution for every lowered
  MMI op (12 random/edge cases each), LQ/SQ, PLZCW, MULT1 and HI1/LO1.
- A standalone PE test runs `SQ → LQ → PADDUB` as native x86-64 and uses
  the guest's `$v0` as the process exit code (42).
- Full suite: 274 tests passing.

## Next

COP1/FPU is the remaining scalar native gap (LWC1/SWC1, arithmetic,
compares and BC1 branches). After that, native codegen covers every scalar
instruction the boot trace reaches; VU0/COP2 remains the only decoder gap.
