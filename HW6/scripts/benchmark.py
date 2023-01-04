#!/usr/bin/env python3


import re
import statistics
import subprocess
from pathlib import Path
from typing import Literal

TIME_PATTERN = re.compile(r"\[conv opencl\].*\[(\d+\.\d+)\] ms")

SRC = Path.home() / "HW6" / "src"
MAKEFILE = SRC / "Makefile"


def compile_program(program: Literal["conv", "cudaconv"]):
    subprocess.run(["make", "-f", MAKEFILE, program])


def run_once(program: Path, filter_idx: Literal[1, 2, 3]) -> float:
    matched = None

    while matched is None:
        output = subprocess.check_output(
            [program, "--filter", str(filter_idx)], encoding="ascii"
        )
        matched = TIME_PATTERN.search(output)

    return float(matched.group(1))


def main() -> None:
    for program in ("conv", "cudaconv"):
        compile_program(program)

        for i in (1, 2, 3):
            times = [run_once(SRC / program, i) for _ in range(15)]
            print(
                f"[{program}]: {statistics.mean(sorted(times)[2:-2]):.4f} ms (filter {i})"
            )


if __name__ == "__main__":
    main()
