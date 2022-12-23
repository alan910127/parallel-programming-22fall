import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence

RUNNER = Path.home() / "HW5" / "scripts" / "run.sh"
TIME_PATTERN = re.compile(r"\[mandelbrot thread\].*\[(\d+\.\d+)\] ms")
RUN_COUNT = 10
VIEW = 1
MAX_ITERATION = 256


def get_kernels():
    for i in range(1, 4):
        yield f"kernel{i}.cu"


def run_task(version: int, view: str = "1", iterations: str = "256") -> float:
    matched = None

    while matched is None:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["sh", RUNNER, str(version), view, iterations],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        matched = TIME_PATTERN.search(result.stdout)

    return float(matched.group(1))


def main(args: Sequence[str]) -> None:
    _, view, iterations, *_ = args
    print(f"{view=}, {iterations=}")
    for k in range(1, 5):
        times: list[float] = []
        for iteration in range(1, RUN_COUNT + 1):
            print(f"\rRunning 'kernel{k}.cu'... {iteration}/{RUN_COUNT}", end="")
            if times:
                print(f" estimated: {statistics.mean(times):.3f} ms", end="")
            elapsed = run_task(k, view, iterations)
            times.append(elapsed)
        print(
            f" (λ ± σ){statistics.mean(times):.3f} ± {statistics.stdev(times):.3f} ms"
        )


if __name__ == "__main__":
    main(sys.argv)
