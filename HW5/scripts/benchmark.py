import re
import statistics
import subprocess
from pathlib import Path

RUNNER = Path.home() / "HW5" / "scripts" / "run.sh"
TIME_PATTERN = re.compile(r"\[mandelbrot thread\].*\[(\d+\.\d+)\] ms")
MAX_ITERATION = 10


def get_kernels():
    for i in range(1, 5):
        yield f"kernel{i}.cu"


def run_task(version: int) -> float:
    matched = None

    while matched is None:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["sh", RUNNER, str(version)], stdout=subprocess.PIPE, encoding="utf-8"
        )
        matched = TIME_PATTERN.search(result.stdout)

    return float(matched.group(1))


def main() -> None:
    for k in range(1, 5):
        times: list[float] = []
        for iteration in range(1, MAX_ITERATION + 1):
            print(f"\rRunning 'kernel{k}.cu'... {iteration}/{MAX_ITERATION}", end="")
            if times:
                print(f" estimated: {statistics.mean(times):.3f} ms", end="")
            elapsed = run_task(k)
            times.append(elapsed)
        print(
            f" (λ ± σ){statistics.mean(times):.3f} ± {statistics.stdev(times):.3f} ms"
        )


if __name__ == "__main__":
    main()
