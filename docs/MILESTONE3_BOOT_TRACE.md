# Boot Trace

- **Source**: `Rumble Racing.cue`
- **Boot ELF**: `SLUS_201.74;1`
- **Entry**: `0x00100008`
- **Steps executed**: 500,000
- **Unique PCs executed**: 606
- **Stop reason**: `step-limit`
- **PC at stop**: `0x00114568`
- **$sp / $gp / $ra / $v0**: `0x1F1610` / `0x1F16F0` / `0x11601C` / `0x0`

## Stop detail

- Step budget exhausted at `0x00114568`; no unsupported instruction or memory fault was reached (the guest is still inside its startup/table-initialisation loops).

## EE syscalls

| Code | Calls |
|---|---|
| `0x3c` | 1 |
| `0x3d` | 1 |
| `0x3e` | 12 |
| `0x64` | 1 |
