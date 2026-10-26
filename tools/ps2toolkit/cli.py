"""ps2toolkit command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_analyze(args) -> int:
    from .analysis.report import analyze_image, render_markdown

    report = analyze_image(args.image)
    md = render_markdown(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"wrote {args.markdown}")
    if not args.json and not args.markdown:
        print(md)
    return 0


def cmd_coverage(args) -> int:
    from .analysis.coverage import analyze_coverage, render_markdown

    report = analyze_coverage(args.image)
    md = render_markdown(report, top=args.top)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"wrote {args.markdown}")
    if not args.json and not args.markdown:
        print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ps2toolkit", description="PS2 disc analysis toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="analyze a disc image (.cue/.bin/.iso)")
    p_analyze.add_argument("image", help="path to .cue, .bin, or .iso")
    p_analyze.add_argument("--json", help="write JSON report to this path")
    p_analyze.add_argument("--markdown", help="write Markdown report to this path")
    p_analyze.set_defaults(func=cmd_analyze)

    p_cov = sub.add_parser(
        "coverage",
        help="instruction coverage of the boot ELF (.cue/.bin/.iso/.elf)")
    p_cov.add_argument("image", help="path to .cue, .bin, .iso, or a boot .elf")
    p_cov.add_argument("--json", help="write JSON report to this path")
    p_cov.add_argument("--markdown", help="write Markdown report to this path")
    p_cov.add_argument("--top", type=int, default=20, help="rows per table (default 20)")
    p_cov.set_defaults(func=cmd_coverage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
