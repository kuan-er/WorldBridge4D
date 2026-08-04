#!/usr/bin/env python3
"""Unattended PRL scheduler for the 3-seed, 3-epoch reliability matrix.

Every train, evaluation, and aggregation remains an independent PRL Run. The
controller serializes launches, explicitly starts the PRL worker when the local
extension leaves a run queued, and waits for terminal run.yaml status.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import threading
import time
import traceback

import yaml

TERMINAL = {"succeeded", "failed", "terminated", "lost"}


class Scheduler:
    def __init__(self, project_root: pathlib.Path, worktree: pathlib.Path, task_id: str,
                 current_full_run: str, dry_run: bool = False):
        self.root = project_root
        self.worktree = worktree
        self.task_id = task_id
        self.current_full_run = current_full_run
        self.dry_run = dry_run
        self.launch_lock = threading.Lock()
        self.child_events = worktree / "configs" / "reliability-child-events.yaml"
        self.node_worker = pathlib.Path("/usr/lib/node_modules/pi-research-loop/dist/cli.js")
        self.summary = {"task_id": task_id, "current_full_run": current_full_run, "runs": [], "errors": []}

    def run_yaml(self, run_id: str) -> pathlib.Path:
        return self.root / "runs" / run_id / "run.yaml"

    def read_run(self, run_id: str) -> dict:
        return yaml.safe_load(self.run_yaml(run_id).read_text())

    def wait_run(self, run_id: str) -> dict:
        while True:
            run = self.read_run(run_id)
            if run["status"] in TERMINAL:
                return run
            time.sleep(20)

    def launch_once(self, name: str, argv: list[str]) -> dict:
        if self.dry_run:
            print(json.dumps({"dry_run": name, "argv": argv}), flush=True)
            return {"run_id": f"DRY-{name}", "status": "succeeded"}
        command = ["prl", "run", "launch", "--task", self.task_id,
                   "--events", str(self.child_events), "--", *argv]
        with self.launch_lock:
            output = subprocess.check_output(command, cwd=self.root, text=True)
        created = json.loads(output)
        run_id = created["run_id"]
        self.summary["runs"].append({"name": name, "run_id": run_id, "argv": argv})
        print(json.dumps({"launched": name, "run_id": run_id, "argv": argv}), flush=True)

        # The server's detached extension worker has previously remained queued.
        # Give it a bounded opportunity to start; otherwise run the official PRL
        # worker in the foreground, which also gives this controller a wait handle.
        time.sleep(5)
        run = self.read_run(run_id)
        if run["status"] == "queued":
            subprocess.run(
                ["node", str(self.node_worker), "internal", "run-worker", run_id],
                cwd=self.root, check=False,
            )
            run = self.read_run(run_id)
        elif run["status"] not in TERMINAL:
            run = self.wait_run(run_id)
        print(json.dumps({"finished": name, "run_id": run_id, "status": run["status"]}), flush=True)
        return run

    def launch(self, name: str, argv: list[str], attempts: int = 2) -> dict:
        last = None
        for attempt in range(1, attempts + 1):
            last = self.launch_once(f"{name}-attempt{attempt}", argv)
            if last["status"] == "succeeded":
                return last
            print(json.dumps({"retrying": name, "attempt": attempt, "status": last["status"]}), flush=True)
        raise RuntimeError(f"{name} failed after {attempts} attempts: {last['status'] if last else 'unknown'}")

    @property
    def stats_cache(self) -> str:
        return "artifacts/reliability/coordinate_stats_train.npz"

    def train(self, mode: str, seed: int) -> dict:
        out = f"artifacts/reliability/seed{seed}/{mode}"
        common = ["scripts/train.py", "--config", f"configs/full_scale_{mode}.yaml",
                  "--seed", str(seed), "--steps", "17211", "--stats-cache", self.stats_cache,
                  "--output-dir", out]
        if mode == "compact":
            argv = ["env", "CUDA_VISIBLE_DEVICES=6", "python", *common]
        else:
            argv = ["env", "CUDA_VISIBLE_DEVICES=0,1", "torchrun", "--standalone",
                    "--nproc_per_node=2", *common]
        return self.launch(f"train-{mode}-seed{seed}", argv)

    def evaluate(self, mode: str, seed: int) -> dict:
        base = f"artifacts/reliability/seed{seed}/{mode}"
        gpu = "6" if mode == "compact" else "0"
        argv = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "python", "scripts/evaluate.py",
                "--checkpoint", f"{base}/checkpoint.pt", "--split", "validation",
                "--pixel-stride", "16", "--query-chunk", "8192",
                "--output", f"{base}/evaluation_validation.json"]
        return self.launch(f"eval-{mode}-seed{seed}", argv)

    def mode_pipeline(self, mode: str):
        try:
            if mode == "full" and not self.dry_run:
                current = self.wait_run(self.current_full_run)
                if current["status"] != "succeeded":
                    raise RuntimeError(f"current Full baseline ended as {current['status']}")
                self.launch(
                    "eval-full-one-epoch",
                    ["env", "CUDA_VISIBLE_DEVICES=0", "python", "scripts/evaluate.py",
                     "--checkpoint", "artifacts/full_scale_full/checkpoint.pt", "--split", "validation",
                     "--pixel-stride", "16", "--query-chunk", "8192",
                     "--output", "artifacts/full_scale_full/evaluation_validation.json"],
                )
            for seed in (2026, 2027, 2028):
                self.train(mode, seed)
                self.evaluate(mode, seed)
            inputs = [f"artifacts/reliability/seed{s}/{mode}/evaluation_validation.json"
                      for s in (2026, 2027, 2028)]
            self.launch(
                f"aggregate-{mode}",
                ["python", "scripts/aggregate_evaluations.py", "--mode", mode,
                 "--inputs", *inputs, "--output", f"artifacts/reliability/{mode}_mean_std.json"],
                attempts=1,
            )
        except Exception as exc:
            self.summary["errors"].append({"mode": mode, "error": repr(exc), "traceback": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    def run(self):
        cache = self.worktree / self.stats_cache
        if not self.dry_run and not cache.exists():
            raise FileNotFoundError(f"Missing coordinate stats cache: {cache}")
        threads = [threading.Thread(target=self.mode_pipeline, args=(mode,), name=mode)
                   for mode in ("compact", "full")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        output = self.worktree / "artifacts" / "reliability" / "scheduler_summary.json"
        if not self.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(self.summary, indent=2))
        print(json.dumps(self.summary, indent=2), flush=True)
        if self.summary["errors"]:
            raise SystemExit(1)
        print("RELIABILITY_MATRIX_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/data/WorldBridge4D")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--current-full-run", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(args.project_root).resolve()
    worktree = pathlib.Path(__file__).resolve().parents[1]
    Scheduler(root, worktree, args.task_id, args.current_full_run, args.dry_run).run()


if __name__ == "__main__":
    main()
