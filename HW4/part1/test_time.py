#!/usr/bin/env python3

import re
import statistics
import subprocess

PROGRAM = "pi_nonblock_linear"

TOSSES = 1000000000
TESTS = 2, 4, 8, 12, 16
EXEC_TIME_RE = re.compile(r"time: (\d+.\d+) Seconds")


def run_once(n_processes: int):
    run = subprocess.run(
        ["make", "run", PROGRAM, str(TOSSES), f"np={n_processes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
    )

    if (match := EXEC_TIME_RE.search(run.stdout)) is not None:
        return float(match.group(1))

    return None


def mean(numbers: list[int]):
    return round(statistics.mean(numbers))


def main():

    for n_processes in TESTS:
        print(f"Running for {n_processes} processes...")
        running_time: list[float] = []

        for _ in range(10):
            if (result := run_once(n_processes)) is None:
                continue
            running_time.append(round(result * 1000))
            print(
                f"\restimated time: {mean(running_time)} ms",
                end="",
            )

        print(
            f"\raverage: {mean(running_time)} ± {statistics.stdev(running_time):.2f} ms"
        )
        print(f"range: {min(running_time)} .. {max(running_time)} ms")


if __name__ == "__main__":
    main()
