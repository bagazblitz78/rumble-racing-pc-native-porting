# Boot ELF Instruction Coverage

- **Source**: `Rumble Racing.cue`
- **Boot ELF**: `SLUS_201.74;1`
- **Entry**: `0x00100008`
- **Executable words**: 244,704
- **Reachable words (recursive descent)**: 155,638
- **Decoded by MVP front-end**: 152,203 (97.79%)
- **Unsupported words**: 3,435
- **Indirect control transfers**: 116

## Executable segments

| Base | Words | Reachable | Unreachable |
|---|---|---|---|
| `0x00100000` | 244,704 | 155,638 | 89,066 |

## Unsupported categories

| Category | Words |
|---|---|
| vu-memory | 1,937 |
| cop2-vu0 | 1,454 |
| 64-bit-memory | 44 |

## Top mnemonics (of 128)

| Mnemonic | Words |
|---|---|
| `addiu` | 20,291 |
| `lw` | 12,163 |
| `nop` | 11,016 |
| `lui` | 7,639 |
| `beq` | 7,444 |
| `sw` | 7,195 |
| `paddub` | 6,889 |
| `jal` | 6,234 |
| `addu` | 5,632 |
| `lwc1` | 4,606 |
| `sq` | 4,399 |
| `sll` | 4,239 |
| `lq` | 4,030 |
| `mtc1` | 3,901 |
| `unsupported` | 3,435 |
| `bne` | 3,318 |
| `ori` | 2,764 |
| `lbu` | 2,673 |
| `sb` | 2,673 |
| `swc1` | 2,625 |

## Unsupported samples

| Address | Word | Category |
|---|---|---|
| `0x001681EC` | `0xD8A90000` | vu-memory |
| `0x001681F4` | `0x48A85800` | cop2-vu0 |
| `0x001681F8` | `0x4BEB4B18` | cop2-vu0 |
| `0x001681FC` | `0xF86C0000` | vu-memory |
| `0x00162A28` | `0xDA090000` | vu-memory |
| `0x00162A2C` | `0x4BC94AAA` | cop2-vu0 |
| `0x00162A30` | `0x4A2B4B3C` | cop2-vu0 |
| `0x00162A34` | `0x4B0A5281` | cop2-vu0 |
| `0x00162A38` | `0x4B0A5282` | cop2-vu0 |
| `0x00162A3C` | `0x4A6A03BE` | cop2-vu0 |
| `0x00162A40` | `0x4A0003BF` | cop2-vu0 |
| `0x00162A44` | `0x4BC04ADC` | cop2-vu0 |
| `0x00162A48` | `0xFA0B0000` | vu-memory |
| `0x0016245C` | `0xD88A0000` | vu-memory |
| `0x00162460` | `0xD8AB0000` | vu-memory |
| `0x00162464` | `0xD8CC0000` | vu-memory |
| `0x00162468` | `0xD8ED0000` | vu-memory |
| `0x0016246C` | `0x4A000D78` | cop2-vu0 |
| `0x00162470` | `0x4A0002FF` | cop2-vu0 |
| `0x0016247C` | `0xF84E0000` | vu-memory |

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

- **VU0/COP2** is the only large undecoded block left (1,937 VU memory forms + 1,454 macro-mode ops), plus 44 unaligned 64-bit forms. The scalar front-end now decodes almost the whole boot path.
- **FPU memory forms** (LWC1/SWC1/LDC1/SDC1) travel with COP1 and need the same FPU state hand-off.
- **64-bit integer forms** are translated as of T3.3: GPRs hold 64-bit values, 32-bit results sign-extend, and DADDU/DSUB/DADDIU, the DSLL/DSRL family and LD/SD/LWU all run natively. Only the unaligned LDL/LDR/SDL/SDR forms remain.
- **COP2/VU0** macro-mode and LQC2/SQC2 need VU0 register state and will likely be stubbed behind an HLE boundary first.
- **Native codegen is now the gap, not decoding.** MMI, LQ/SQ and COP1 run in the golden model but still need SSE2 lowering in x64emit before the recompiler can execute the same code.
- **Indirect control transfers** shown above are jump tables and function pointers; static recompilation must resolve them (or fall back to a dispatch stub) before the whole image is reachable.
