# Boot ELF Instruction Coverage

- **Source**: `Rumble Racing.cue`
- **Boot ELF**: `SLUS_201.74;1`
- **Entry**: `0x00100008`
- **Executable words**: 244,704
- **Reachable words (recursive descent)**: 155,613
- **Decoded by MVP front-end**: 130,882 (84.11%)
- **Unsupported words**: 24,731
- **Indirect control transfers**: 116

## Executable segments

| Base | Words | Reachable | Unreachable |
|---|---|---|---|
| `0x00100000` | 244,704 | 155,613 | 89,091 |

## Unsupported categories

| Category | Words |
|---|---|
| cop1-fpu | 14,065 |
| fpu-memory | 7,231 |
| vu-memory | 1,937 |
| cop2-vu0 | 1,454 |
| 64-bit-memory | 44 |

## Top mnemonics (of 101)

| Mnemonic | Words |
|---|---|
| `unsupported` | 24,731 |
| `addiu` | 20,291 |
| `lw` | 12,163 |
| `nop` | 11,016 |
| `lui` | 7,635 |
| `beq` | 7,444 |
| `sw` | 7,195 |
| `paddub` | 6,889 |
| `jal` | 6,234 |
| `addu` | 5,632 |
| `sq` | 4,399 |
| `sll` | 4,239 |
| `lq` | 4,030 |
| `bne` | 3,317 |
| `ori` | 2,761 |
| `lbu` | 2,673 |
| `sb` | 2,673 |
| `sd` | 2,168 |
| `or` | 1,734 |
| `andi` | 1,640 |

## Unsupported samples

| Address | Word | Category |
|---|---|---|
| `0x001681B0` | `0xC4600060` | fpu-memory |
| `0x001681C8` | `0xE48025D0` | fpu-memory |
| `0x001681D0` | `0xC48025D0` | fpu-memory |
| `0x001681DC` | `0x44830800` | cop1-fpu |
| `0x001681E4` | `0x46000802` | cop1-fpu |
| `0x001681E8` | `0xE48025D4` | fpu-memory |
| `0x001681EC` | `0xD8A90000` | vu-memory |
| `0x001681F0` | `0x44080800` | cop1-fpu |
| `0x001681F4` | `0x48A85800` | cop2-vu0 |
| `0x001681F8` | `0x4BEB4B18` | cop2-vu0 |
| `0x001681FC` | `0xF86C0000` | vu-memory |
| `0x0016FCB0` | `0x44826000` | cop1-fpu |
| `0x0016FCBC` | `0x44826800` | cop1-fpu |
| `0x00170440` | `0x44830000` | cop1-fpu |
| `0x00170444` | `0xC4A10094` | fpu-memory |
| `0x00170448` | `0x468000E0` | cop1-fpu |
| `0x00170450` | `0xC4A000AC` | fpu-memory |
| `0x00170454` | `0x46800860` | cop1-fpu |
| `0x00170458` | `0x44831000` | cop1-fpu |
| `0x00170460` | `0x46031019` | cop1-fpu |

## Indirect control transfers

| Address | Word | Kind | Register |
|---|---|---|---|
| `0x001498C8` | `0x0040F809` | indirect-call | `$2` |
| `0x00127BB8` | `0x0040F809` | indirect-call | `$2` |
| `0x0011B6C0` | `0x00800008` | indirect-jump | `$4` |
| `0x001183D4` | `0x0060F809` | indirect-call | `$3` |
| `0x001184F4` | `0x0040F809` | indirect-call | `$2` |
| `0x00118634` | `0x0040F809` | indirect-call | `$2` |
| `0x00117B94` | `0x0040F809` | indirect-call | `$2` |
| `0x0011873C` | `0x0280F809` | indirect-call | `$20` |
| `0x00100E2C` | `0x00800008` | indirect-jump | `$4` |
| `0x0012CB10` | `0x0040F809` | indirect-call | `$2` |
| `0x0016D324` | `0x00400008` | indirect-jump | `$2` |
| `0x0016C174` | `0x0040F809` | indirect-call | `$2` |
| `0x0016C1C0` | `0x0040F809` | indirect-call | `$2` |
| `0x001A71E8` | `0x0040F809` | indirect-call | `$2` |
| `0x001C5D74` | `0x0040F809` | indirect-call | `$2` |
| `0x001C5DE0` | `0x0040F809` | indirect-call | `$2` |
| `0x00145DA4` | `0x0040F809` | indirect-call | `$2` |
| `0x001BFD9C` | `0x0040F809` | indirect-call | `$2` |
| `0x001BF644` | `0x0040F809` | indirect-call | `$2` |
| `0x001BF654` | `0x0040F809` | indirect-call | `$2` |

_(96 more not shown)_

## Implications for the recompiler

- **COP1/FPU** now dominates the remaining gap (14,065 scalar words plus 7,231 FPU memory forms). LQ/SQ, COP0 and the MMI set used by the game are decoded and executed by the golden model as of T3.5, but the x64 emitter still needs SSE2 lowering for MMI/LQ/SQ.
- **FPU memory forms** (LWC1/SWC1/LDC1/SDC1) travel with COP1 and need the same FPU state hand-off.
- **64-bit integer forms** are translated as of T3.3: GPRs hold 64-bit values, 32-bit results sign-extend, and DADDU/DSUB/DADDIU, the DSLL/DSRL family and LD/SD/LWU all run natively. Only the unaligned LDL/LDR/SDL/SDR forms remain.
- **COP2/VU0** macro-mode and LQC2/SQC2 need VU0 register state and will likely be stubbed behind an HLE boundary first.
- **Cheap wins already landed** (T3.2): MOVZ/MOVN, SYNC, CACHE/PREF and the branch-likely / REGIMM branch-and-link forms. The residual 'other' bucket is small enough to inspect case by case.
- **Indirect control transfers** shown above are jump tables and function pointers; static recompilation must resolve them (or fall back to a dispatch stub) before the whole image is reachable.
