#!/usr/bin/env python3
"""Best-effort, ordered checkpoint replication from a fast local tier.

The source is opened before taking the destination lock, so pruning its pathname
cannot invalidate an in-flight copy. Replicas serialize under an advisory lock.
A requested-step reservation prevents an older delayed process from replacing a
newer durable checkpoint. Any failure leaves the previous destination intact.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from typing import BinaryIO


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_open_file(reader: BinaryIO, destination: Path, expected_size: int,
                   chunk_bytes: int, throttle_seconds: float) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.replica.tmp"
    )
    temporary.unlink(missing_ok=True)
    written = 0
    try:
        with temporary.open("wb") as writer:
            while block := reader.read(chunk_bytes):
                writer.write(block)
                written += len(block)
                if throttle_seconds:
                    time.sleep(throttle_seconds)
            writer.flush()
            os.fsync(writer.fileno())
        if written != expected_size:
            raise OSError(
                f"checkpoint replica size mismatch: wrote={written}, expected={expected_size}"
            )
        temporary.replace(destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def replicate_checkpoint(source: Path, destination: Path, step: int,
                         chunk_bytes: int = 16 << 20,
                         throttle_seconds: float = 0.02) -> str:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".replica.lock")
    status_path = destination.with_suffix(destination.suffix + ".replica.json")

    # Opening first pins the inode even if the trainer prunes the source name.
    with source.open("rb") as reader, lock_path.open("a+b") as lock:
        expected_size = os.fstat(reader.fileno()).st_size
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status: dict[str, object] = {}
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                status = {}
        requested = int(status.get("requested_step", -1))
        if requested >= step:
            return "superseded"

        # Reserve this step before the long copy. If copying fails, an older
        # delayed worker still cannot downgrade the durable destination.
        atomic_json(status_path, {
            "requested_step": int(step),
            "completed_step": int(status.get("completed_step", -1)),
            "source": str(source),
            "destination": str(destination),
            "state": "copying",
            "pid": os.getpid(),
        })
        copy_open_file(
            reader, destination, expected_size, int(chunk_bytes),
            float(throttle_seconds),
        )
        atomic_json(status_path, {
            "requested_step": int(step),
            "completed_step": int(step),
            "source": str(source),
            "destination": str(destination),
            "bytes": expected_size,
            "state": "complete",
            "pid": os.getpid(),
        })
        return "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--chunk-bytes", type=int, default=16 << 20)
    parser.add_argument("--throttle-seconds", type=float, default=0.02)
    args = parser.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    started = time.monotonic()
    try:
        result = replicate_checkpoint(
            Path(args.source), Path(args.destination), args.step,
            args.chunk_bytes, args.throttle_seconds,
        )
        print(json.dumps({
            "event": "durable_checkpoint_replica", "result": result,
            "step": args.step, "source": args.source,
            "destination": args.destination,
            "elapsed_seconds": time.monotonic() - started,
        }), flush=True)
    except Exception as error:
        # This detached best-effort process must never signal failure to the
        # trainer. The previous durable checkpoint remains atomically intact.
        print(json.dumps({
            "event": "durable_checkpoint_replica_failed", "step": args.step,
            "source": args.source, "destination": args.destination,
            "error": repr(error), "elapsed_seconds": time.monotonic() - started,
        }), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
