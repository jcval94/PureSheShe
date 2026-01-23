from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Union
import csv
import json
import re
import time

import tracemalloc
import cProfile
import pstats


@dataclass
class ProfilingConfig:
    enabled: bool = False
    cpu: bool = True
    memory: bool = True
    output_dir: Union[str, Path] = "experiments_outputs"
    label: str = "profile"
    top_n: int = 50


def _slug(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "profile"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


@contextmanager
def profile_context(name: str, config: Optional[ProfilingConfig]) -> Iterator[None]:
    if config is None or not config.enabled:
        yield
        return

    output_dir = Path(config.output_dir)
    label = _slug(config.label)
    run_name = _slug(name)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{label}_{run_name}_{timestamp}"
    profiler = cProfile.Profile() if config.cpu else None

    if config.memory:
        tracemalloc.start()

    if profiler is not None:
        profiler.enable()

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if profiler is not None:
            profiler.disable()

        cpu_rows: list[dict] = []
        cpu_summary: dict = {"elapsed_s": elapsed, "top_n": int(config.top_n)}
        if profiler is not None:
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            items = []
            for (filename, line, func), (cc, nc, tt, ct, callers) in stats.stats.items():
                items.append(
                    {
                        "function": func,
                        "filename": filename,
                        "line": line,
                        "callcount": nc,
                        "primitive_calls": cc,
                        "tottime": tt,
                        "cumtime": ct,
                    }
                )
            items.sort(key=lambda row: row["cumtime"], reverse=True)
            cpu_rows = items[: int(config.top_n)]
            cpu_summary.update(
                {
                    "total_time_s": stats.total_tt,
                    "total_calls": stats.total_calls,
                }
            )

        if cpu_rows:
            cpu_csv = Path(f"{base}.cpu.csv")
            cpu_json = Path(f"{base}.cpu.json")
            _write_csv(cpu_csv, cpu_rows)
            _write_json(
                cpu_json,
                {"summary": cpu_summary, "stats": cpu_rows},
            )

        if config.memory:
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            mem_rows = []
            for stat in snapshot.statistics("lineno")[: int(config.top_n)]:
                frame = stat.traceback[0]
                mem_rows.append(
                    {
                        "filename": frame.filename,
                        "line": frame.lineno,
                        "size_bytes": stat.size,
                        "count": stat.count,
                    }
                )
            if mem_rows:
                mem_csv = Path(f"{base}.memory.csv")
                mem_json = Path(f"{base}.memory.json")
                _write_csv(mem_csv, mem_rows)
                _write_json(
                    mem_json,
                    {
                        "summary": {"elapsed_s": elapsed, "top_n": int(config.top_n)},
                        "stats": mem_rows,
                    },
                )
