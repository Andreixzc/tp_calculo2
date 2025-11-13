#!/usr/bin/env python3
"""
Cross-platform runner to render all Manim animation files in this repo.

Usage examples:
  python render_all.py                # uses default pattern '*_animation.py' and quality 'qh'
  python render_all.py -p '*.py' -q ql -w 4
  python render_all.py --dry-run      # show commands but don't run

This script uses the active Python interpreter (sys.executable) to invoke Manim
via `python -m manim` so it will use the same virtualenv if you run the script
with the project's venv activated or by calling the venv's python explicitly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path
from typing import List


def find_files(pattern: str, root: Path) -> List[Path]:
    files = sorted(root.glob(pattern))
    # ignore this runner file
    files = [f for f in files if f.name != Path(__file__).name]
    return files


def build_command(python_exe: str, file: Path, quality: str) -> List[str]:
    # Use `python -m manim` to ensure manim from the chosen python is used
    # Use the `-a` flag to render all scenes in the file (matches project's render_all.sh)
    cmd = [python_exe, "-m", "manim", f"-{quality}", "-a", str(file)]
    return cmd


def run_one(cmd: List[str], dry_run: bool = False) -> int:
    print("\n📹 Running:", " ".join(cmd))
    if dry_run:
        return 0
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print("❌ Error running command:", e)
        return 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render all manim animation files in this folder.")
    parser.add_argument(
        "-p",
        "--pattern",
        default="*_animation.py",
        help="glob pattern to find animation files (default: '*_animation.py')",
    )
    parser.add_argument("-q", "--quality", default="qh", help="manim quality flag (e.g. ql, qm, qh)")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="number of parallel workers to run manim processes (default 1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    args = parser.parse_args(argv)

    root = Path(__file__).parent.resolve()
    files = find_files(args.pattern, root)
    if not files:
        print(f"No files matched pattern '{args.pattern}' in {root}")
        return 2

    print(f"Found {len(files)} files to render (pattern: {args.pattern}).")
    python_exe = sys.executable
    print(f"Using python: {python_exe}")

    commands = [build_command(python_exe, f, args.quality) for f in files]

    # Run sequentially or in a thread pool depending on workers
    failures: List[tuple[Path, int]] = []

    if args.workers == 1:
        for file, cmd in zip(files, commands):
            rc = run_one(cmd, dry_run=args.dry_run)
            if rc != 0:
                failures.append((file, rc))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_file = {ex.submit(run_one, cmd, args.dry_run): file for file, cmd in zip(files, commands)}
            for fut in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[fut]
                try:
                    rc = fut.result()
                except Exception as e:
                    print(f"❌ Exception rendering {file}: {e}")
                    failures.append((file, 1))
                else:
                    if rc != 0:
                        failures.append((file, rc))

    print("\n--- Summary ---")
    if not failures:
        print("✅ All animations rendered (or dry-run).")
        print("📁 media/videos/")
        return 0
    else:
        print(f"❌ {len(failures)} files failed:")
        for f, rc in failures:
            print(f" - {f}  (rc={rc})")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
