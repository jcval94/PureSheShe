"""Utilities for running DelDel experiments on the corner dataset.

The helpers in this module orchestrate the DelDel boundary discovery pipeline
over a configurable grid of parameters. Every run records execution time,
collects statistics about the discovered planes, and aggregates per-dimension
class distributions suitable for quick inspection, e.g.::

    for i in range(1, 5):
        print([x["target_class"] for x in valuable_["by_dim"][i]])

"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .datasets import make_corner_class_dataset
from .engine import DelDel, DelDelConfig, DeltaRecord
from .frontier_planes_all_modes import compute_frontier_planes_all_modes


@dataclass
class PlaneEntry:
    """Summary for a single frontier plane discovered in an experiment run."""

    pair: Tuple[int, int]
    label: int
    plane_index: int
    dominant_dim: int
    target_class: Optional[int]
    record_indices: List[int]
    record_count: int
    plane_score: float
    coverage: float
    planarity: float
    fit_error: Mapping[str, float]


@dataclass
class ExperimentRun:
    """Aggregated information for a single parameter configuration."""

    params: Mapping[str, Any]
    runtime_s: float
    total_pairs: int
    total_planes: int
    errors: List[str] = field(default_factory=list)
    plane_entries: List[PlaneEntry] = field(default_factory=list)

    @property
    def by_dim_counts(self) -> Dict[int, Counter]:
        counts: Dict[int, Counter] = defaultdict(Counter)
        for entry in self.plane_entries:
            key = entry.target_class if entry.target_class is not None else "unassigned"
            counts[entry.dominant_dim][key] += 1
        return counts

    @property
    def avg_planarity(self) -> float:
        if not self.plane_entries:
            return 0.0
        return float(np.mean([p.planarity for p in self.plane_entries]))

    @property
    def avg_coverage(self) -> float:
        if not self.plane_entries:
            return 0.0
        return float(np.mean([p.coverage for p in self.plane_entries]))


def _majority_class(records: Sequence[DeltaRecord], indices: Iterable[int]) -> Optional[int]:
    votes: Counter = Counter()
    for idx in indices:
        try:
            rec = records[int(idx)]
        except (IndexError, TypeError):
            continue
        cls = getattr(rec, "y1", None)
        if cls is None:
            cls = getattr(rec, "y0", None)
        if cls is None:
            continue
        votes[int(cls)] += 1
    if not votes:
        return None
    return int(votes.most_common(1)[0][0])


def _collect_plane_entries(
    res: Mapping[Tuple[int, int], Mapping[str, Any]],
    records: Sequence[DeltaRecord],
    *,
    n_dims: int,
) -> Tuple[List[PlaneEntry], List[str]]:
    entries: List[PlaneEntry] = []
    errors: List[str] = []

    for pair, block in res.items():
        if block.get("error"):
            errors.append(f"pair {pair}: {block['error']}")
        planes_by_label = block.get("planes_by_label", {}) or {}
        assignment = block.get("assignment", {}) or {}
        rec_indices = np.asarray(assignment.get("rec_indices", []), dtype=int)
        assigned_label = np.asarray(assignment.get("assigned_label", []), dtype=int)
        assigned_plane = np.asarray(assignment.get("assigned_plane", []), dtype=int)

        for label, planes in planes_by_label.items():
            for plane_idx, plane in enumerate(planes):
                mask = (
                    (assigned_label == int(label))
                    & (assigned_plane == int(plane_idx))
                )
                assigned = rec_indices[mask] if mask.any() else np.empty(0, dtype=int)
                dominant_dim = int(np.argmax(np.abs(np.asarray(plane["n"], float))))
                dominant_dim = int(np.clip(dominant_dim + 1, 1, n_dims))
                tgt_class = _majority_class(records, assigned)
                entry = PlaneEntry(
                    pair=tuple(map(int, pair)),
                    label=int(label),
                    plane_index=int(plane_idx),
                    dominant_dim=dominant_dim,
                    target_class=tgt_class,
                    record_indices=assigned.tolist(),
                    record_count=int(assigned.size),
                    plane_score=float(plane.get("score", 0.0)),
                    coverage=float(plane.get("coverage", 0.0)),
                    planarity=float(plane.get("planarity", 0.0)),
                    fit_error={k: float(v) for k, v in (plane.get("fit_error", {}) or {}).items()},
                )
                entries.append(entry)
    return entries, errors


def _default_param_grid() -> List[Dict[str, Any]]:
    return [
        dict(min_cluster_size=2, max_models_per_round=3, max_depth=1),
        dict(min_cluster_size=3, max_models_per_round=3, max_depth=1, angle_merge_deg=5.5, offset_merge_tau=0.015),
        dict(min_cluster_size=4, max_models_per_round=4, max_depth=2, angle_merge_deg=8.0),
        dict(min_cluster_size=5, max_models_per_round=5, max_depth=3, angle_merge_deg=9.5, offset_merge_tau=0.035, seed=7),
        dict(min_cluster_size=6, max_models_per_round=6, max_depth=3, angle_merge_deg=12.0, offset_merge_tau=0.05, mode="VMF", prefer_cp=False),
        dict(min_cluster_size=7, max_models_per_round=6, max_depth=3, angle_merge_deg=14.0, offset_merge_tau=0.06, mode="OFFSETS", seed=13),
    ]


def run_corner_pipeline_experiments(
    *,
    param_grid: Optional[Sequence[Mapping[str, Any]]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    deldel_config: Optional[DelDelConfig] = None,
    csv_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the DelDel pipeline for several parameter combinations.

    Parameters
    ----------
    param_grid:
        Sequence with dictionaries describing overrides for
        :func:`compute_frontier_planes_all_modes`. If omitted a small default
        grid is used.
    dataset_kwargs:
        Optional keyword arguments forwarded to
        :func:`make_corner_class_dataset`.
    deldel_config:
        Optional :class:`DelDelConfig` instance used to generate flip records.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing dataset/model summaries, timing metrics, the
        list of experiment runs, and the aggregated ``valuable_`` structure.
    """

    dataset_kwargs = dict(dataset_kwargs or {})
    X, y, feature_names = make_corner_class_dataset(**dataset_kwargs)
    n_samples, n_dims = X.shape
    class_counts = Counter(map(int, y))

    # Train a simple multinomial logistic regression model once for all runs.
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        multi_class="auto",
        max_iter=600,
        solver="lbfgs",
        random_state=dataset_kwargs.get("random_state", 0),
    )
    t0 = perf_counter()
    model.fit(X, y)
    model_time = perf_counter() - t0

    cfg = deldel_config or DelDelConfig(
        segments_target=6,
        near_frac=0.6,
        k_near_base=2,
        k_far_per_i=1,
        q_near=0.2,
        q_far=0.8,
        margin_quantile=0.2,
        secant_iters=1,
        final_bisect=2,
        min_logit_gain=0.0,
        min_pair_margin_end=0.0,
        prob_swing_weight=0.5,
        use_jsd=False,
        random_state=dataset_kwargs.get("random_state", 0),
    )

    engine = DelDel(cfg)
    t0 = perf_counter()
    engine.fit(X, model)
    fit_time = perf_counter() - t0
    records = list(engine.records_)

    grid = list(param_grid) if param_grid is not None else _default_param_grid()
    runs: List[ExperimentRun] = []

    for params in grid:
        call_kwargs = dict(params)
        start = perf_counter()
        res = compute_frontier_planes_all_modes(records, **call_kwargs)
        runtime = perf_counter() - start

        entries, errors = _collect_plane_entries(res, records, n_dims=n_dims)
        total_planes = sum(len(lst) for block in res.values() for lst in (block.get("planes_by_label", {}) or {}).values())
        run = ExperimentRun(
            params=dict(params),
            runtime_s=runtime,
            total_pairs=len(res),
            total_planes=int(total_planes),
            errors=errors,
            plane_entries=entries,
        )
        runs.append(run)

    valuable_by_dim: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(1, n_dims + 1)}
    plane_rows: List[Dict[str, Any]] = []
    dim_distribution_rows: List[Dict[str, Any]] = []

    for idx, run in enumerate(runs):
        for entry in run.plane_entries:
            valuable_by_dim[entry.dominant_dim].append(
                dict(
                    run_index=idx,
                    params=dict(run.params),
                    pair=entry.pair,
                    label=entry.label,
                    plane_index=entry.plane_index,
                    target_class=entry.target_class,
                    record_count=entry.record_count,
                    coverage=entry.coverage,
                    planarity=entry.planarity,
                )
            )
            plane_rows.append(
                dict(
                    run_index=idx,
                    pair_a=entry.pair[0],
                    pair_b=entry.pair[1],
                    label=entry.label,
                    plane_index=entry.plane_index,
                    dominant_dim=entry.dominant_dim,
                    target_class=entry.target_class if entry.target_class is not None else "",
                    record_count=entry.record_count,
                    coverage=entry.coverage,
                    planarity=entry.planarity,
                    plane_score=entry.plane_score,
                    fit_error=";".join(
                        f"{k}:{v:.6f}" for k, v in sorted(entry.fit_error.items())
                    ),
                )
            )

        counts = run.by_dim_counts
        for dim in range(1, n_dims + 1):
            if isinstance(counts, dict):
                dim_counter = counts.get(dim, Counter())
            else:
                dim_counter = counts[dim]
            total = float(sum(dim_counter.values()))
            if dim_counter:
                for cls, cnt in sorted(dim_counter.items()):
                    dim_distribution_rows.append(
                        dict(
                            run_index=idx,
                            dim=dim,
                            target_class=cls,
                            plane_count=cnt,
                            proportion=(cnt / total) if total else 0.0,
                        )
                    )
            else:
                dim_distribution_rows.append(
                    dict(
                        run_index=idx,
                        dim=dim,
                        target_class="",
                        plane_count=0,
                        proportion=0.0,
                    )
                )

    csv_outputs: Dict[str, str] = {}
    if csv_dir is not None:
        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "experiments_summary.csv"
        with summary_path.open("w", newline="") as fh:
            fieldnames = [
                "run_index",
                "runtime_s",
                "total_pairs",
                "total_planes",
                "avg_planarity",
                "avg_coverage",
                "errors",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for idx, run in enumerate(runs):
                writer.writerow(
                    dict(
                        run_index=idx,
                        runtime_s=f"{run.runtime_s:.6f}",
                        total_pairs=run.total_pairs,
                        total_planes=run.total_planes,
                        avg_planarity=f"{run.avg_planarity:.6f}",
                        avg_coverage=f"{run.avg_coverage:.6f}",
                        errors=";".join(run.errors),
                    )
                )
        csv_outputs["experiments"] = str(summary_path)

        planes_path = out_dir / "plane_entries.csv"
        with planes_path.open("w", newline="") as fh:
            if plane_rows:
                fieldnames = list(plane_rows[0].keys())
            else:
                fieldnames = [
                    "run_index",
                    "pair_a",
                    "pair_b",
                    "label",
                    "plane_index",
                    "dominant_dim",
                    "target_class",
                    "record_count",
                    "coverage",
                    "planarity",
                    "plane_score",
                    "fit_error",
                ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in plane_rows:
                writer.writerow(row)
        csv_outputs["planes"] = str(planes_path)

        dim_path = out_dir / "dimension_distributions.csv"
        with dim_path.open("w", newline="") as fh:
            fieldnames = ["run_index", "dim", "target_class", "plane_count", "proportion"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in dim_distribution_rows:
                writer.writerow(row)
        csv_outputs["dimension_distributions"] = str(dim_path)

    summary = dict(
        dataset=dict(
            n_samples=int(n_samples),
            n_dims=int(n_dims),
            feature_names=list(feature_names),
            class_counts=dict(class_counts),
        ),
        timings=dict(
            model_fit_s=model_time,
            deldel_fit_s=fit_time,
        ),
        experiments=[
            dict(
                params=dict(run.params),
                runtime_s=run.runtime_s,
                total_pairs=run.total_pairs,
                total_planes=run.total_planes,
                avg_planarity=run.avg_planarity,
                avg_coverage=run.avg_coverage,
                by_dim={dim: dict(counter) for dim, counter in run.by_dim_counts.items()},
                errors=list(run.errors),
            )
            for run in runs
        ],
        valuable_=dict(by_dim=valuable_by_dim),
        records_total=len(records),
        csv_outputs=csv_outputs,
    )
    return summary
