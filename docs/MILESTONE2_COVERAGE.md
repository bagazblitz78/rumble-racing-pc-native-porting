# Boot ELF Instruction Coverage

- **Source**: `Rumble Racing.cue`
- **Boot ELF**: `SLUS_201.74;1`
- **Entry**: `0x00100008`
- **Executable words**: 244,704
- **Reachable words (recursive descent)**: 155,235
- **Decoded by MVP front-end**: 106,933 (68.88%)
- **Unsupported words**: 48,302
- **Indirect control transfers**: 116

## Executable segments

| Base | Words | Reachable | Unreachable |
|---|---|---|---|
| `0x00100000` | 244,704 | 155,235 | 89,469 |

## Unsupported categories

| Category | Words |
|---|---|
| mmi | 15,805 |
| cop1-fpu | 14,000 |
| fpu-memory | 7,231 |
| 64-bit-memory | 3,717 |
| vu-memory | 1,937 |
| 64-bit-shift | 1,856 |
| cop2-vu0 | 1,454 |
| 64-bit-alu | 1,134 |
| other | 832 |
| conditional-move | 152 |
| cop0 | 79 |
| sync | 49 |
| regimm | 42 |
| cache-pref | 14 |

## Top mnemonics (of 56)

| Mnemonic | Words |
|---|---|
| `unsupported` | 48,302 |
| `addiu` | 20,253 |
| `lw` | 12,138 |
| `nop` | 11,005 |
| `lui` | 7,599 |
| `beq` | 7,422 |
| `sw` | 7,180 |
| `jal` | 6,234 |
| `addu` | 5,627 |
| `sll` | 4,235 |
| `bne` | 3,299 |
| `ori` | 2,746 |
| `lbu` | 2,673 |
| `sb` | 2,666 |
| `or` | 1,730 |
| `andi` | 1,636 |
| `jr` | 1,572 |
| `slti` | 1,071 |
| `subu` | 829 |
| `lb` | 815 |

## Unsupported samples

| Address | Word | Category |
|---|---|---|
| `0x00100020` | `0x7C400000` | mmi |
| `0x0010005C` | `0x0080E02D` | 64-bit-alu |
| `0x00100068` | `0x0040E82D` | 64-bit-alu |
| `0x0010008C` | `0x42000038` | cop0 |
| `0x001000A8` | `0x0040202D` | 64-bit-alu |
| `0x00121624` | `0x7FBF0020` | mmi |
| `0x00121628` | `0x7FB10010` | mmi |
| `0x0012162C` | `0x7FB00000` | mmi |
| `0x00121630` | `0x70808E28` | mmi |
| `0x0012163C` | `0x72202628` | mmi |
| `0x00121650` | `0x7BBF0020` | mmi |
| `0x00121654` | `0x7BB10010` | mmi |
| `0x00121658` | `0x7BB00000` | mmi |
| `0x0012165C` | `0x70001628` | mmi |
| `0x001215B4` | `0x7FBF0000` | mmi |
| `0x00121610` | `0x7BBF0000` | mmi |
| `0x00164174` | `0x7FBF0000` | mmi |
| `0x001641A0` | `0x7BBF0000` | mmi |
| `0x001202D4` | `0x7FBF0020` | mmi |
| `0x001202D8` | `0x7FB10010` | mmi |

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

- **MMI + COP1** dominate the gap. The R5900 multimedia instructions and the FPU are the first large extensions required; scalar integer code alone cannot execute the boot ELF.
- **FPU memory forms** (LWC1/SWC1/LDC1/SDC1) travel with COP1 and need the same FPU state hand-off.
- **64-bit integer forms** (DADDU/DSUB/DADDI/LD/SD and the DSLL/DSRL family) require 64-bit GPR semantics; the current MVP stores 32-bit values, so this is an architectural change, not a decoder table entry.
- **COP2/VU0** macro-mode and LQC2/SQC2 need VU0 register state and will likely be stubbed behind an HLE boundary first.
- **Cheap wins** before the big blocks: MOVZ/MOVN, SYNC, CACHE/PREF (no-ops), and the REGIMM branch forms (BLTZAL/BGEZAL/BLTZL/BGEZL).
- **Indirect control transfers** shown above are jump tables and function pointers; static recompilation must resolve them (or fall back to a dispatch stub) before the whole image is reachable.
