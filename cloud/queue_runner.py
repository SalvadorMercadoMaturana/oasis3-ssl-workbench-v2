#!/usr/bin/env python
"""Run the lines of a job file ``P`` at a time; lines appended later are picked up too.

    nohup python cloud/queue_runner.py /root/runs/jobs.txt --parallel 10 &

Each line is ``<name> <train.py or script arguments ...>`` (``#`` comments and blank
lines skipped); a line starting with ``script:`` runs ``python <rest>`` instead of
``python train.py``. Logs go to ``<jobs dir>/logs/<name>.log`` and every finished job
appends ``<name> exit <code>`` to ``logs/finished.txt``. The runner keeps polling the job
file until ``<jobs dir>/STOP`` exists and nothing is left, then writes ``QUEUE_DONE``.
Restarting it skips the names already in ``finished.txt``. No torch import.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def read_jobs(path: Path) -> list[tuple[str, list[str]]]:
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, *rest = shlex.split(line)
        jobs.append((name, rest))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--poll", type=float, default=20.0)
    args = ap.parse_args()
    base = args.jobs.parent
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    finished_file = logs / "finished.txt"
    done = set()
    if finished_file.is_file():
        done = {ln.split()[0] for ln in finished_file.read_text().splitlines() if ln.strip()}
    running: dict[str, subprocess.Popen] = {}
    started: set[str] = set(done)
    while True:
        for name, proc in list(running.items()):
            code = proc.poll()
            if code is not None:
                with finished_file.open("a") as f:
                    f.write(f"{name} exit {code}\n")
                del running[name]
        pending = [(n, a) for n, a in read_jobs(args.jobs) if n not in started]
        while pending and len(running) < args.parallel:
            name, rest = pending.pop(0)
            if rest and rest[0].startswith("script:"):
                cmd = [sys.executable, rest[0][len("script:"):], *rest[1:]]
            else:
                cmd = [sys.executable, "train.py", *rest]
            log = (logs / f"{name}.log").open("w")
            running[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            started.add(name)
        if not running and not pending and (base / "STOP").exists():
            (base / "QUEUE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
