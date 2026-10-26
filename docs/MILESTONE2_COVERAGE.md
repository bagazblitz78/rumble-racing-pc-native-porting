# Boot ELF Instruction Coverage

- **Source**: `Rumble Racing.cue`
- **Boot ELF**: `SLUS_201.74;1`
- **Entry**: `0x00100008`
- **Executable words**: 244,704
- **Reachable words (recursive descent)**: 155,613
- **Decoded by MVP front-end**: 123,502 (79.36%)
- **Unsupported words**: 32,111
- **Indirect control transfers**: 116

## Executable segments

| Base | Words | Reachable | Unreachable |
|---|---|---|---|
| `0x00100000` | 244,704 | 155,613 | 89,091 |

## Unsupported categories

| Category | Words |
|---|---|
| cop1-fpu | 14,065 |
| mmi | 7,380 |
| fpu-memory | 7,231 |
| vu-memory | 1,937 |
| cop2-vu0 | 1,454 |
| 64-bit-memory | 44 |

## Top mnemonics (of 86)

| Mnemonic | Words |
|---|---|
| `unsupported` | 32,111 |
| `addiu` | 20,291 |
| `lw` | 12,163 |
| `nop` | 11,016 |
| `lui` | 7,635 |
| `beq` | 7,444 |
| `sw` | 7,195 |
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
| `jr` | 1,589 |

## Unsupported samples

| Address | Word | Category |
|---|---|---|
| `0x00121630` | `0x70808E28` | mmi |
| `0x0012163C` | `0x72202628` | mmi |
| `0x0012165C` | `0x70001628` | mmi |
| `0x0011DDBC` | `0x70C53018` | mmi |
| `0x001ABFC0` | `0x70808628` | mmi |
| `0x001AC028` | `0x72202628` | mmi |
| `0x001AC038` | `0x72201628` | mmi |
| `0x001209EC` | `0x70802E28` | mmi |
| `0x00120A28` | `0x700846E9` | mmi |
| `0x00120AA8` | `0x70801628` | mmi |
| `0x00120A30` | `0x71081389` | mmi |
| `0x001209E4` | `0x70801628` | mmi |
| `0x001209C0` | `0x70802E28` | mmi |
| `0x001AC030` | `0x70002E28` | mmi |
| `0x001AC010` | `0x70001628` | mmi |
| `0x001B6270` | `0x70002628` | mmi |
| `0x00122980` | `0x70008628` | mmi |
| `0x00122990` | `0x72002628` | mmi |
| `0x0010EA94` | `0x70E31818` | mmi |
| `0x00122998` | `0x70002E28` | mmi |

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

- **MMI + COP1** dominate the remaining gap. LQ/SQ are decoded and executed by the golden model as of T3.4, so the next vector extension is the MMI opcode space; the FPU is larger overall and equally needed before native code can run the game.
- **FPU memory forms** (LWC1/SWC1/LDC1/SDC1) travel with COP1 and need the same FPU state hand-off.
- **64-bit integer forms** are translated as of T3.3: GPRs hold 64-bit values, 32-bit results sign-extend, and DADDU/DSUB/DADDIU, the DSLL/DSRL family and LD/SD/LWU all run natively. Only the unaligned LDL/LDR/SDL/SDR forms remain.
- **COP2/VU0** macro-mode and LQC2/SQC2 need VU0 register state and will likely be stubbed behind an HLE boundary first.
- **Cheap wins already landed** (T3.2): MOVZ/MOVN, SYNC, CACHE/PREF and the branch-likely / REGIMM branch-and-link forms. The residual 'other' bucket is small enough to inspect case by case.
- **Indirect control transfers** shown above are jump tables and function pointers; static recompilation must resolve them (or fall back to a dispatch stub) before the whole image is reachable.
