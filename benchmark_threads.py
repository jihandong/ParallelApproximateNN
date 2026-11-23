import argparse
import re
import subprocess
from pathlib import Path

from matplotlib.figure import Figure


STAGES = ("Train", "Add", "Search")
TIMINGS = re.compile(
    r"^Train: (\S+) seconds\nAdd: (\S+) seconds\n\d+ queries in (\S+) seconds$",
    re.MULTILINE,
)
DEBUG_TIMINGS = re.compile(
    r"^(K-means\+\+|Lloyd|IVF Add) profiling \(ms\): (.+)$", re.MULTILINE
)


def profiling_times(output):
    profiles = dict(DEBUG_TIMINGS.findall(output))
    if len(profiles) != 3:
        raise SystemExit("Rebuild ann in Debug to include Train and Add profiling.")
    parts = {}
    for stage, fields in profiles.items():
        for field in fields.split(", "):
            name, milliseconds = field.split("=", 1)
            parts[f"{stage} {name}"] = float(milliseconds)
    return parts


def main():
    parser = argparse.ArgumentParser(description="Compare ANN stage times across thread counts.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--max-threads", type=int, required=True)
    args = parser.parse_args()
    if args.max_threads < 1:
        parser.error("--max-threads must be positive")

    exe = args.executable.resolve()
    dataset = args.dataset.resolve()
    thread_counts = range(1, args.max_threads + 1)
    times = {"faiss": [], "customized": []}
    for threads in thread_counts:
        for impl in times:
            label = f"{impl}, {threads} thread(s)"
            print(label, flush=True)
            run = subprocess.run(
                [str(exe), str(dataset), "--impl", impl, "--threads", str(threads)],
                capture_output=True,
                text=True,
            )
            match = TIMINGS.search(run.stdout)
            if run.returncode or match is None:
                raise SystemExit(f"Benchmark failed for {label}\n{run.stderr}{run.stdout}")
            stage_times = tuple(map(float, match.groups()))
            times[impl].append(stage_times)
            if impl == "customized":
                parts = profiling_times(run.stderr + "\n" + run.stdout)
                parts["Search"] = stage_times[2] * 1000
                parts["Others"] = max(0.0, sum(stage_times) * 1000 - sum(parts.values()))

    comparison = Figure(figsize=(12, 4), constrained_layout=True)
    for index, (ax, stage) in enumerate(zip(comparison.subplots(1, 3), STAGES)):
        for impl, runs in times.items():
            ax.plot(thread_counts, [run[index] for run in runs], marker="o", label=impl)
        ax.set(title=stage, xlabel="Threads", ylabel="Seconds")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(alpha=0.25)
        ax.legend()
    comparison.suptitle(dataset.name)
    comparison.savefig(f"{dataset.name}-threads.png", dpi=150)

    breakdown = Figure(figsize=(8, 4), constrained_layout=True)
    ax = breakdown.subplots()
    total = sum(parts.values())
    wedges, _ = ax.pie(list(parts.values()), startangle=90)
    labels = [f"{name}: {ms / total:.2%}" for name, ms in parts.items()]
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    breakdown.suptitle(f"{dataset.name}: customized profiled time ({args.max_threads} threads)")
    breakdown.savefig(f"{dataset.name}-customized-pie.png", dpi=150)


if __name__ == "__main__":
    main()
