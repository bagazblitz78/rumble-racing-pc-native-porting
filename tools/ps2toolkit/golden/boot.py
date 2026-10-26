"""Boot tracer: execute a real PS2 ELF entry under the golden model (T3.4).

This is the first harness that runs the game's own code instead of
synthetic test programs. It loads the PT_LOAD segments into the 32 MiB
guest RAM, starts at `e_entry`, and steps until:

- the decoder hits an instruction the front-end does not know yet,
- a memory access leaves RAM,
- a `break` instruction executes,
- the guest reaches pc == 0, or
- the step budget runs out.

EE kernel `syscall`s are dispatched through a simple HLE table: by default
the call is assumed to return 0 in `$v0` and execution continues at
pc + 4. This is enough to walk the C runtime's init prologue (RPC setup,
heap init, constructors) and find the first *real* blocker, which is what
drives the next instruction-set milestone.

The returned report is JSON-ready; the live `EEMemory` is kept alongside
it so tests can inspect what the boot code wrote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..elf.ps2elf import PS2ELF, PT_LOAD
from .interpreter import (
    EEMemory,
    EEState,
    GuestBreak,
    GuestSyscall,
    MemoryFault,
    UnsupportedInstruction,
    step,
)


@dataclass
class TraceResult:
    report: dict
    memory: EEMemory


def load_guest_image(elf: PS2ELF) -> EEMemory:
    """Copy every PT_LOAD segment into a fresh 32 MiB guest RAM."""
    mem = EEMemory()
    for seg in elf.segments:
        if seg.type != PT_LOAD or seg.filesz == 0:
            continue
        payload = elf.data[seg.offset : seg.offset + seg.filesz]
        if len(payload) != seg.filesz:
            raise ValueError("segment data truncated")
        mem.write(seg.vaddr, payload)
    return mem


def _regs_summary(st: EEState) -> dict:
    return {
        "v0": st.regs[2],
        "sp": st.regs[29],
        "gp": st.regs[28],
        "ra": st.regs[31],
        "pc": st.pc,
    }


SyscallHook = Callable[[int, EEState, EEMemory], int]


def trace_boot(
    elf: PS2ELF,
    max_steps: int = 200_000,
    syscall_returns: dict[int, int] | SyscallHook | None = None,
    exit_syscalls: set[int] | None = None,
    max_syscall_samples: int = 32,
) -> TraceResult:
    """Run `elf` from its entry point and report where it stopped.

    `syscall_returns` is either a {number: $v0 result} map or a
    `hook(number, state, memory) -> $v0` callable. `exit_syscalls` stop the
    trace as if the guest terminated.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    mem = load_guest_image(elf)
    st = EEState(pc=elf.entry)
    returns = syscall_returns
    hook: SyscallHook | None = returns if callable(returns) else None
    ret_table = {} if hook is not None else dict(returns or {})
    exits = set(exit_syscalls or ())

    steps = 0
    unique_pcs: set[int] = set()
    syscalls: dict[int, int] = {}
    syscall_log: list[dict] = []
    stop_reason = "step-limit"
    detail: dict | None = None

    while steps < max_steps:
        if st.pc == 0:
            stop_reason = "pc-zero"
            break
        unique_pcs.add(st.pc)
        try:
            steps += step(st, mem)
        except GuestSyscall as exc:
            number = exc.number if exc.code == 0 else exc.code
            syscalls[number] = syscalls.get(number, 0) + 1
            if len(syscall_log) < max_syscall_samples:
                syscall_log.append({"number": number, "code": exc.code,
                                    "pc": exc.pc})
            if number in exits:
                stop_reason = "exit-syscall"
                detail = {"syscall": number, "pc": exc.pc}
                break
            if hook is not None:
                st._set(2, hook(number, st, mem))
            else:
                st._set(2, ret_table.get(number, 0))
            st.pc = exc.pc + 4
            steps += 1
        except UnsupportedInstruction as exc:
            stop_reason = "unsupported"
            detail = {
                "pc": exc.addr,
                "word": exc.raw,
                "mnemonic": exc.mnemonic,
            }
            break
        except MemoryFault as exc:
            stop_reason = "memory-fault"
            detail = {"pc": st.pc, "message": str(exc)}
            break
        except GuestBreak as exc:
            stop_reason = "break"
            detail = {"pc": exc.pc, "code": exc.code}
            break
    if stop_reason == "step-limit":
        detail = {"pc": st.pc}

    report = {
        "entry": elf.entry,
        "steps": steps,
        "unique_pcs": len(unique_pcs),
        "stop_reason": stop_reason,
        "detail": detail,
        "pc": st.pc,
        "registers": {k: v & 0xFFFFFFFFFFFFFFFF for k, v in _regs_summary(st).items()},
        "mem_write_checksum": sum(mem.data) & 0xFFFFFFFF,
        "syscalls": {f"0x{code:02x}": n for code, n in sorted(syscalls.items())},
        "syscall_log": syscall_log,
        "hi": st.hi,
        "lo": st.lo,
    }
    return TraceResult(report=report, memory=mem)


def render_markdown(report: dict, source: str = "", elf_name: str = "") -> str:
    lines = [
        "# Boot Trace",
        "",
        f"- **Source**: `{source}`",
        f"- **Boot ELF**: `{elf_name}`",
        f"- **Entry**: `0x{report['entry']:08X}`",
        f"- **Steps executed**: {report['steps']:,}",
        f"- **Unique PCs executed**: {report['unique_pcs']:,}",
        f"- **Stop reason**: `{report['stop_reason']}`",
        f"- **PC at stop**: `0x{report['pc']:08X}`",
        f"- **$sp / $gp / $ra / $v0**: "
        f"`0x{report['registers']['sp']:X}` / `0x{report['registers']['gp']:X}` / "
        f"`0x{report['registers']['ra']:X}` / `0x{report['registers']['v0']:X}`",
        "",
        "## Stop detail",
        "",
    ]
    detail = report["detail"]
    if not detail:
        lines.append("None.")
    elif report["stop_reason"] == "unsupported":
        lines.append(
            f"- Unsupported `{detail['mnemonic']}` word `0x{detail['word']:08X}` "
            f"at `0x{detail['pc']:08X}`"
        )
    elif report["stop_reason"] == "memory-fault":
        lines.append(f"- At `0x{detail['pc']:08X}`: {detail['message']}")
    elif report["stop_reason"] == "break":
        lines.append(f"- `break` code {detail['code']} at `0x{detail['pc']:08X}`")
    elif report["stop_reason"] == "step-limit":
        lines.append(f"- Step budget exhausted at `0x{detail['pc']:08X}`")
    else:
        lines.append(f"- `{detail}`")

    lines += ["", "## EE syscalls", "", "| Code | Calls |", "|---|---|"]
    if report["syscalls"]:
        for code, n in report["syscalls"].items():
            lines.append(f"| `{code}` | {n:,} |")
    else:
        lines.append("| _none_ | 0 |")
    return "\n".join(lines) + "\n"
