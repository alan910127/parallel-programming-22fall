import argparse
import re
import statistics
import subprocess
import sys


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=10,
        help="number of iterations (default=10)",
    )
    parser.add_argument(
        "--testcase",
        "-t",
        choices=("1", "2", "3"),
        default="1",
        help="which test case to run (default=1)",
    )
    parser.add_argument(
        "--make-args",
        "-m",
        dest="make_args",
        nargs="*",
        default=[],
        help="arguments taken by make",
    )
    return parser


def main() -> None:

    parser = configure_parser()
    args = parser.parse_args()

    # rebuilt project
    cleanup = subprocess.run(
        ["make", "clean"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if cleanup.returncode != 0:
        print("Error occurred while cleaning up", file=sys.stderr)
        exit(1)

    compilation = subprocess.run(
        ["make", *args.make_args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if compilation.returncode != 0:
        print("Error occurred while compiling", file=sys.stderr)
        exit(1)

    time_pattern = re.compile(r"(\d+.\d+)sec")
    elapsed_times: list[float] = []

    for test_round in range(args.num):
        print(f"Running test #{test_round + 1:3d}... ", end="", flush=True)

        completed = subprocess.run(
            ["./test_auto_vectorize", "-t", args.testcase],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )

        if (matched := time_pattern.search(completed.stdout)) is not None:
            seconds = float(matched.group(1))
            print(f"{seconds} seconds")
            elapsed_times.append(seconds)

    print(f"median of time: {statistics.median(elapsed_times)} seconds")
    print(f"average time: {statistics.mean(elapsed_times)} seconds")


if __name__ == "__main__":
    main()
