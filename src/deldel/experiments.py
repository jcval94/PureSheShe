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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import csv
import json
import itertools
import logging
from collections import Counter, defaultdict
from pathlib import Path
import warnings

import numpy as np

from deldel.utils import _verbosity_to_level

from .datasets import make_corner_class_dataset
from .engine import ChangePointConfig, DelDel, DelDelConfig, DeltaRecord
from .frontier_planes_all_modes import compute_frontier_planes_all_modes
from .find_low_dim_spaces_fast import find_low_dim_spaces
from .globalmaj import prune_and_orient_planes_unified_globalmaj


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


def _write_stage_timings_csv(
    stage_timings: Sequence[Mapping[str, Any]], fh: Any
) -> None:
    """Serialize stage timing dictionaries to a CSV file."""

    base_fields = ["stage", "callable", "duration_s"]
    extra_fields: List[str] = []
    for entry in stage_timings:
        for key in entry.keys():
            if key in base_fields or key in extra_fields:
                continue
            extra_fields.append(key)

    fieldnames = base_fields + extra_fields
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()

    for entry in stage_timings:
        row = {key: "" for key in fieldnames}
        row["stage"] = entry.get("stage", "")
        row["callable"] = entry.get("callable", "")

        duration = entry.get("duration_s", 0.0)
        try:
            row["duration_s"] = f"{float(duration):.6f}"
        except (TypeError, ValueError):
            row["duration_s"] = ""

        for key in extra_fields:
            value = entry.get(key, "")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = f"{float(value):.6f}"
            else:
                row[key] = value

        writer.writerow(row)


def _setup_logger(name: str, verbosity: int) -> logging.Logger:
    level = _verbosity_to_level(verbosity)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logger


def _stage_entry(
    *,
    stage: str,
    callable: str,
    start_s: float,
    end_s: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = dict(
        stage=stage,
        callable=callable,
        start_s=start_s,
        end_s=end_s,
        duration_s=end_s - start_s,
    )
    if extra:
        entry.update(extra)
    return entry


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
    verbosity: int = 0,
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

    logger = _setup_logger(__name__, verbosity)
    level = _verbosity_to_level(verbosity)

    dataset_kwargs = dict(dataset_kwargs or {})
    logger.log(level, "Preparando dataset corner con parámetros: %s", dataset_kwargs)
    try:
        X, y, feature_names = make_corner_class_dataset(
            **dataset_kwargs, verbosity=max(verbosity - 1, 0)
        )
    except TypeError:
        logger.log(level, "make_corner_class_dataset no acepta verbosity, reintentando")
        X, y, feature_names = make_corner_class_dataset(**dataset_kwargs)
    except Exception:
        logger.exception("Fallo al generar el dataset corner")
        raise
    n_samples, n_dims = X.shape
    class_counts = Counter(map(int, y))
    logger.log(level, "Dataset generado: n=%d dims=%d clases=%s", n_samples, n_dims, class_counts)

    # Train a simple multinomial logistic regression model once for all runs.
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        random_state=dataset_kwargs.get("random_state", 0),
    )
    t0 = perf_counter()
    try:
        model.fit(X, y)
    except Exception:
        logger.exception("Fallo entrenando LogisticRegression")
        raise
    model_time = perf_counter() - t0
    logger.log(level, "LogisticRegression entrenado en %.4f s", model_time)

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
    try:
        engine.fit(X, model)
    except Exception:
        logger.exception("Error en DelDel.fit")
        raise
    fit_time = perf_counter() - t0
    records = list(engine.records_)
    logger.log(
        level,
        "DelDel completado en %.4f s | registros=%d",
        fit_time,
        len(records),
    )

    grid = list(param_grid) if param_grid is not None else _default_param_grid()
    runs: List[ExperimentRun] = []

    for params in grid:
        call_kwargs = dict(params)
        call_kwargs["verbosity"] = max(verbosity - 1, 0)
        logger.log(level, "Ejecutando frontier con parámetros: %s", call_kwargs)
        start = perf_counter()
        try:
            res = compute_frontier_planes_all_modes(records, **call_kwargs)
        except Exception:
            logger.exception("compute_frontier_planes_all_modes falló")
            raise
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
        logger.log(
            level,
            "Frontier terminada en %.4f s | pares=%d planos=%d",
            runtime,
            len(res),
            total_planes,
        )

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


def _build_selection_from_frontier(
    frontier: Mapping[Tuple[int, int], Mapping[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Transform the frontier payload into a selection usable by
    :func:`find_low_dim_spaces`.

    The helper keeps only the information required by
    :func:`find_low_dim_spaces`, namely the winning planes for each class
    pair.  The orientation for each plane is inferred by evaluating which
    side of the inequality better covers the corresponding class.
    """

    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    classes = sorted(np.unique(y))

    by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    winning_planes: List[Dict[str, Any]] = []
    per_class_regions: Dict[int, List[str]] = {int(cls): [] for cls in classes}

    Xa_by_cls = {int(cls): X[y == int(cls)] for cls in classes}

    plane_counter = 0
    for pair, payload in frontier.items():
        a, b = map(int, pair)
        planes_payload = payload.get("planes_by_label", {}) or {}
        pair_planes: List[Dict[str, Any]] = []
        for planes in planes_payload.values():
            for plane in planes or []:
                n = np.asarray(plane.get("n"), float).reshape(-1)
                if n.size != X.shape[1]:
                    continue
                b0 = float(plane.get("b", 0.0))
                plane_id = plane.get("plane_id") or f"pl_{a}_{b}_{plane_counter:04d}"
                plane_counter += 1

                Xa = Xa_by_cls.get(a, np.empty((0, X.shape[1])))
                Xb = Xa_by_cls.get(b, np.empty((0, X.shape[1])))

                def _score(side: int) -> float:
                    if side >= 0:
                        mask_a = (Xa @ n + b0) <= 0.0
                        mask_b = (Xb @ n + b0) >= 0.0
                    else:
                        mask_a = (Xa @ n + b0) >= 0.0
                        mask_b = (Xb @ n + b0) <= 0.0
                    score_a = float(mask_a.mean()) if mask_a.size else 0.0
                    score_b = float(mask_b.mean()) if mask_b.size else 0.0
                    return score_a + score_b

                side = +1 if _score(+1) >= _score(-1) else -1

                entry = dict(
                    plane_id=plane_id,
                    origin_pair=(a, b),
                    n=n.tolist(),
                    b=b0,
                    side=int(side),
                )
                pair_planes.append(dict(entry))
                winning_planes.append(dict(entry))

        if pair_planes:
            by_pair[(a, b)] = dict(
                winning_planes=pair_planes,
                meta=dict(num_candidates=len(pair_planes)),
            )

    return dict(
        by_pair_augmented=by_pair,
        winning_planes=winning_planes,
        regions_global=dict(per_plane=[], per_class=per_class_regions),
        meta=dict(feature_names=list(feature_names)),
    )


def _build_demo_selection(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
) -> Dict[str, Any]:
    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    classes = sorted(np.unique(y))
    per_class_regions: Dict[int, List[str]] = {int(c): [] for c in classes}
    per_plane: List[Dict[str, Any]] = []
    by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    winning_planes: List[Dict[str, Any]] = []

    mu = {int(c): X[y == int(c)].mean(axis=0) for c in classes}
    pid_counter = 0
    for a, b in itertools.combinations(classes, 2):
        diff = mu[int(b)] - mu[int(a)]
        if not np.any(np.isfinite(diff)):
            continue
        dim = int(np.argmax(np.abs(diff)))
        thr = float(0.5 * (mu[int(a)][dim] + mu[int(b)][dim]))
        n = np.zeros(X.shape[1], dtype=float)
        n[dim] = 1.0
        b0 = -thr
        if mu[int(a)][dim] <= mu[int(b)][dim]:
            side_ab = +1
        else:
            side_ab = -1

        plane_id = f"demo_pl{pid_counter:04d}"
        pid_counter += 1
        oriented_id = f"{plane_id}:{'≤' if side_ab >= 0 else '≥'}"
        region_low_id = f"demo_rg_{plane_id}_c{int(a)}"
        region_high_id = f"demo_rg_{plane_id}_c{int(b)}"

        winning = dict(
            plane_id=plane_id,
            origin_pair=(int(a), int(b)),
            n=n.tolist(),
            b=b0,
            side=side_ab,
            _pid=pid_counter,
            global_for=int(a),
            regions_global=[region_low_id, region_high_id],
        )
        winning_planes.append(winning)

        region_low = dict(
            region_id=region_low_id,
            plane_id=plane_id,
            oriented_plane_id=oriented_id,
            class_id=int(a),
            side=side_ab,
            geometry=dict(n=n.tolist(), b=b0, side=side_ab),
            origin_pair=(int(a), int(b)),
        )
        per_plane.append(region_low)
        per_class_regions[int(a)].append(region_low_id)

        region_high = dict(
            region_id=region_high_id,
            plane_id=plane_id,
            oriented_plane_id=oriented_id,
            class_id=int(b),
            side=-side_ab,
            geometry=dict(n=n.tolist(), b=b0, side=-side_ab),
            origin_pair=(int(a), int(b)),
        )
        per_plane.append(region_high)
        per_class_regions[int(b)].append(region_high_id)

        by_pair[(int(a), int(b))] = dict(
            winning_planes=[winning],
            other_planes=[],
            region_rule=dict(
                logic="AND",
                dims=(dim,),
                inequalities=[
                    dict(pid=pid_counter, n=n.tolist(), b=b0, side="≤" if side_ab >= 0 else "≥")
                ],
            ),
            metrics_overall={},
            meta=dict(num_candidates=1),
        )

    return dict(
        by_pair_augmented=by_pair,
        winning_planes=winning_planes,
        regions_global=dict(per_plane=per_plane, per_class=per_class_regions),
        meta=dict(feature_names=list(feature_names)),
    )


def run_corner_random_forest_pipeline(
    *,
    random_state: int = 0,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    deldel_config: Optional[DelDelConfig] = None,
    cp_config: Optional[ChangePointConfig] = None,
    frontier_kwargs: Optional[Mapping[str, Any]] = None,
    finder_param_grid: Optional[Sequence[Mapping[str, Any]]] = None,
    csv_dir: Optional[Union[str, Path]] = None,
    dataset_builder: Optional[Callable[..., Tuple[np.ndarray, np.ndarray, Sequence[str]]]] = None,
    verbosity: int = 0,
) -> Dict[str, Any]:
    """Execute the primary DelDel pipeline on the corner dataset.

    The routine mirrors the reference pipeline shared by the maintainers: it
    generates the canonical synthetic dataset via
    :func:`make_corner_class_dataset`, fits a compact
    :class:`~sklearn.ensemble.RandomForestClassifier`, extracts DelDel flip
    records with an aggressive change-point configuration, and evaluates a grid
    of :func:`find_low_dim_spaces` parameters.  Stage runtimes and finder
    outcomes can be persisted as CSV files for offline inspection.
    """

    from sklearn.ensemble import RandomForestClassifier

    logger = _setup_logger(__name__, verbosity)
    level = _verbosity_to_level(verbosity)
    stage_timings: List[Dict[str, Any]] = []

    dataset_kwargs = dict(dataset_kwargs or {})
    builder = dataset_builder or make_corner_class_dataset
    dataset_name = getattr(builder, "__name__", str(builder))
    if builder is make_corner_class_dataset:
        dataset_kwargs.setdefault("n_per_cluster", 180)
        dataset_kwargs.setdefault("std_class1", 0.5)
        dataset_kwargs.setdefault("std_other", 0.85)
        dataset_kwargs.setdefault("a", 2.8)
        dataset_kwargs.setdefault("random_state", random_state)

    ds_start = perf_counter()
    try:
        X, y, feature_names = builder(
            **dataset_kwargs, verbosity=max(verbosity - 1, 0)
        )
    except TypeError:
        logger.log(level, "%s no acepta verbosity, reintentando sin el parámetro", dataset_name)
        X, y, feature_names = builder(**dataset_kwargs)
    except Exception:
        logger.exception("Error generando dataset con %s", dataset_name)
        raise
    ds_end = perf_counter()
    class_counts = {int(cls): int((y == int(cls)).sum()) for cls in np.unique(y)}
    stage_timings.append(
        _stage_entry(
            stage="dataset_generation",
            callable=dataset_name,
            start_s=ds_start,
            end_s=ds_end,
            extra=dict(
                params=json.dumps(dataset_kwargs, sort_keys=True),
                n_samples=int(X.shape[0]),
                n_features=int(X.shape[1]),
                classes=len(class_counts),
                class_distribution=json.dumps(class_counts, sort_keys=True),
            ),
        )
    )
    logger.log(
        level,
        "Dataset listo: %d muestras, %d variables, clases=%s",
        X.shape[0],
        X.shape[1],
        class_counts,
    )

    model = RandomForestClassifier(n_estimators=30, random_state=random_state)

    model_start = perf_counter()
    try:
        model.fit(X, y)
    except Exception:
        logger.exception("Fallo al entrenar el RandomForest base")
        raise
    model_end = perf_counter()
    stage_timings.append(
        _stage_entry(
            stage="model_fit",
            callable="sklearn.ensemble.RandomForestClassifier.fit",
            start_s=model_start,
            end_s=model_end,
            extra=dict(
                n_estimators=30,
                depth=model.max_depth,
                samples_used=int(X.shape[0]),
                features_used=int(X.shape[1]),
            ),
        )
    )
    logger.log(level, "Modelo base entrenado en %.3f s", model_end - model_start)

    cfg = deldel_config or DelDelConfig(
        segments_target=750,
        random_state=random_state,
        log_level=level,
    )
    cp_cfg = cp_config or ChangePointConfig(
        enabled=True,
        mode="treefast",
        per_record_max_points=4,
        max_candidates=128,
        max_bisect_iters=6,
    )

    engine = DelDel(cfg, cp_cfg)
    deldel_start = perf_counter()
    try:
        engine.fit(X, model, verbose=verbosity > 1)
    except Exception:
        logger.exception("Error durante DelDel.fit")
        raise
    deldel_end = perf_counter()
    stage_timings.append(
        _stage_entry(
            stage="deldel_fit",
            callable="deldel.engine.DelDel.fit",
            start_s=deldel_start,
            end_s=deldel_end,
            extra=dict(
                records_generated=len(getattr(engine, "records_", [])),
                cp_enabled=cp_cfg.enabled,
                segments_target=cfg.segments_target,
            ),
        )
    )
    records = list(engine.records_)
    success_records = sum(1 for r in records if getattr(r, "success", True))
    logger.log(
        level,
        "DelDel.fit completado en %.3f s | registros=%d (éxito=%d)",
        deldel_end - deldel_start,
        len(records),
        success_records,
    )

    pair_start = perf_counter()
    pairs = sorted(
        {
            (
                int(getattr(rec, "y0", -1)),
                int(getattr(rec, "y1", -1)),
            )
            for rec in records
            if getattr(rec, "success", True)
            and getattr(rec, "y0", None) is not None
            and getattr(rec, "y1", None) is not None
            and int(rec.y0) != int(rec.y1)
        }
    )
    pair_end = perf_counter()
    stage_timings.append(
        _stage_entry(
            stage="pair_identification",
            callable="deldel.experiments.run_corner_random_forest_pipeline/pairs",
            start_s=pair_start,
            end_s=pair_end,
            extra=dict(
                pairs_total=len(pairs),
                records_total=len(records),
                successful_records=success_records,
            ),
        )
    )
    logger.log(level, "Pares de clase detectados: %d", len(pairs))

    frontier_kwargs = dict(frontier_kwargs or {})
    frontier_kwargs.setdefault("mode", "C")
    frontier_kwargs.setdefault("seed", random_state)
    frontier_kwargs.setdefault("verbosity", max(verbosity - 1, 0))

    frontier_start = perf_counter()
    try:
        frontier = compute_frontier_planes_all_modes(
            records,
            pairs=pairs,
            **frontier_kwargs,
        )
    except Exception:
        logger.exception("Fallo en compute_frontier_planes_all_modes")
        raise
    frontier_end = perf_counter()
    frontier_plane_count = sum(
        len(lst)
        for block in frontier.values()
        for lst in (block.get("planes_by_label", {}) or {}).values()
    )
    stage_timings.append(
        _stage_entry(
            stage="frontier_computation",
            callable="deldel.frontier_planes_all_modes.compute_frontier_planes_all_modes",
            start_s=frontier_start,
            end_s=frontier_end,
            extra=dict(
                pairs_total=len(pairs),
                planes_total=int(frontier_plane_count),
                frontier_params=json.dumps(frontier_kwargs, sort_keys=True),
            ),
        )
    )
    logger.log(
        level,
        "Fronteras calculadas en %.3f s | pares=%d, planos=%d",
        frontier_end - frontier_start,
        len(pairs),
        frontier_plane_count,
    )

    prune_start = perf_counter()
    try:
        selection = prune_and_orient_planes_unified_globalmaj(
            frontier,
            X,
            y,
            max_k=10,
            min_improve=1e-3,
            feature_names=feature_names,
            dims_for_text=(0, 1),
            min_region_size=10,
            min_abs_diff=0.02,
            min_rel_lift=0.05,
            verbosity=max(verbosity - 1, 0),
        )
    except Exception:
        logger.exception("Fallo en prune_and_orient_planes_unified_globalmaj")
        raise
    prune_end = perf_counter()
    stage_timings.append(
        _stage_entry(
            stage="selection_prune",
            callable="deldel.globalmaj.prune_and_orient_planes_unified_globalmaj",
            start_s=prune_start,
            end_s=prune_end,
            extra=dict(
                winning_planes=len(selection.get("winning_planes", [])),
                regions_total=len(selection.get("regions_global", {}).get("per_plane", [])),
            ),
        )
    )
    logger.log(
        level,
        "Poda y orientación completadas en %.3f s | planos ganadores=%d",
        prune_end - prune_start,
        len(selection.get("winning_planes", [])),
    )

    finder_param_grid = list(finder_param_grid) if finder_param_grid is not None else [
        dict(
            max_planes_in_rule=3,
            max_planes_per_pair=4,
            min_support=40,
            min_rel_gain_f1=0.05,
            min_lift_prec=1.40,
            consider_dims_up_to=X.shape[1],
            rng_seed=random_state,
        ),
    ]

    finder_runs: List[Dict[str, Any]] = []
    n_dims = X.shape[1]

    for idx, params in enumerate(finder_param_grid):
        params = dict(params)
        params.setdefault("feature_names", feature_names)
        params.setdefault("max_planes_in_rule", 3)
        params.setdefault("max_planes_per_pair", 4)
        params.setdefault("min_support", 40)
        params.setdefault("min_rel_gain_f1", 0.05)
        params.setdefault("min_lift_prec", 1.40)
        params.setdefault("consider_dims_up_to", n_dims)
        params.setdefault("rng_seed", random_state)

        feature_names_param = params.pop("feature_names")

        t_run = perf_counter()
        try:
            valuable = find_low_dim_spaces(
                X,
                y,
                selection,
                feature_names=feature_names_param,
                verbosity=max(verbosity - 1, 0),
                **params,
            )
        except Exception:
            logger.exception("find_low_dim_spaces falló para iter %d", idx)
            raise
        runtime = perf_counter() - t_run

        counts_by_dim = {dim: len(valuable.get(dim, [])) for dim in range(1, n_dims + 1)}
        best_f1 = {}
        best_precision = {}
        for dim, regions in valuable.items():
            if not regions:
                best_f1[dim] = 0.0
                best_precision[dim] = 0.0
                continue
            best_f1[dim] = max(float(r.get("metrics", {}).get("f1", 0.0)) for r in regions)
            best_precision[dim] = max(
                float(r.get("metrics", {}).get("precision", 0.0)) for r in regions
            )

        stage_timings.append(
            _stage_entry(
                stage=f"find_low_dim_spaces_run_{idx}",
                callable="deldel.find_low_dim_spaces_fast.find_low_dim_spaces",
                start_s=t_run,
                end_s=t_run + runtime,
                extra=dict(
                    run_index=idx,
                    params=json.dumps(params, sort_keys=True),
                    regions_total=sum(len(valuable.get(dim, [])) for dim in valuable),
                    best_f1=json.dumps(best_f1, sort_keys=True),
                    best_precision=json.dumps(best_precision, sort_keys=True),
                ),
            )
        )

        finder_runs.append(
            dict(
                run_index=idx,
                runtime_s=runtime,
                params=params,
                regions_total=sum(counts_by_dim.values()),
                regions_by_dim=counts_by_dim,
                best_f1_by_dim=best_f1,
                best_precision_by_dim=best_precision,
            )
        )

        logger.log(
            level,
            "Finder iter %d completado en %.3f s | regiones=%d | best_f1=%s | best_prec=%s",
            idx,
            runtime,
            sum(counts_by_dim.values()),
            best_f1,
            best_precision,
        )

    csv_outputs: Dict[str, str] = {}
    if csv_dir is not None:
        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stages_path = out_dir / "stage_timings.csv"
        with stages_path.open("w", newline="") as fh:
            _write_stage_timings_csv(stage_timings, fh)
        csv_outputs["stage_timings"] = str(stages_path)

        finder_path = out_dir / "finder_runs.csv"
        fieldnames = [
            "run_index",
            "runtime_s",
            "params_json",
            "regions_total",
        ]
        for dim in range(1, n_dims + 1):
            fieldnames.append(f"regions_dim{dim}")
            fieldnames.append(f"best_f1_dim{dim}")
            fieldnames.append(f"best_precision_dim{dim}")

        with finder_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in finder_runs:
                row = dict(
                    run_index=entry["run_index"],
                    runtime_s=f"{entry['runtime_s']:.6f}",
                    params_json=json.dumps(entry["params"], sort_keys=True),
                    regions_total=entry["regions_total"],
                )
                for dim in range(1, n_dims + 1):
                    row[f"regions_dim{dim}"] = entry["regions_by_dim"].get(dim, 0)
                    row[f"best_f1_dim{dim}"] = (
                        f"{entry['best_f1_by_dim'].get(dim, 0.0):.6f}"
                    )
                    row[f"best_precision_dim{dim}"] = (
                        f"{entry['best_precision_by_dim'].get(dim, 0.0):.6f}"
                    )
                writer.writerow(row)
        csv_outputs["finder_runs"] = str(finder_path)

    summary = dict(
        dataset=dict(
            name=dataset_name,
            n_samples=int(X.shape[0]),
            n_dims=int(X.shape[1]),
            feature_names=feature_names,
            class_counts={int(cls): int((y == int(cls)).sum()) for cls in np.unique(y)},
            params=dataset_kwargs,
        ),
        model=dict(
            estimator="RandomForestClassifier",
            n_estimators=int(model.n_estimators),
            random_state=random_state,
        ),
        deldel_config=cfg,
        change_point_config=cp_cfg,
        stage_timings=stage_timings,
        frontier_pairs=len(pairs),
        finder_runs=finder_runs,
        csv_outputs=csv_outputs,
    )
    return summary


def run_iris_random_forest_pipeline(**kwargs: Any) -> Dict[str, Any]:
    """Backward compatible alias relying on the corner dataset.

    The Iris-based pipeline has been retired to align tests, experiments and
    documentation around :func:`make_corner_class_dataset`.  The alias forwards
    arguments to :func:`run_corner_random_forest_pipeline` and emits a
    ``DeprecationWarning``.
    """

    warnings.warn(
        "run_iris_random_forest_pipeline has been replaced by "
        "run_corner_random_forest_pipeline and now uses "
        "make_corner_class_dataset.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_corner_random_forest_pipeline(**kwargs)


def run_corner_pipeline_with_low_dim(
    *,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    deldel_config: Optional[DelDelConfig] = None,
    frontier_kwargs: Optional[Mapping[str, Any]] = None,
    finder_param_grid: Optional[Sequence[Mapping[str, Any]]] = None,
    csv_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run the complete DelDel pipeline and evaluate low dimensional regions.

    The experiment generates the synthetic corner dataset, trains a logistic
    regression model, extracts frontier planes with :class:`DelDel`, and
    finally explores low dimensional spaces with several configurations of
    :func:`find_low_dim_spaces`.  Execution times for every stage and the
    finder configurations are collected and optionally persisted as CSV
    files.
    """

    dataset_kwargs = dict(dataset_kwargs or {})
    dataset_kwargs.setdefault("n_per_cluster", 180)
    dataset_kwargs.setdefault("std_class1", 0.55)
    dataset_kwargs.setdefault("std_other", 0.9)
    dataset_kwargs.setdefault("a", 2.6)
    dataset_kwargs.setdefault("random_state", 0)

    stage_timings: List[Dict[str, Any]] = []

    t0 = perf_counter()
    X, y, feature_names = make_corner_class_dataset(**dataset_kwargs)
    stage_timings.append(
        dict(
            stage="dataset_generation",
            callable="deldel.datasets.make_corner_class_dataset",
            duration_s=perf_counter() - t0,
        )
    )

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        random_state=dataset_kwargs.get("random_state", 0),
    )

    t0 = perf_counter()
    model.fit(X, y)
    stage_timings.append(
        dict(
            stage="model_fit",
            callable="sklearn.linear_model.LogisticRegression.fit",
            duration_s=perf_counter() - t0,
        )
    )

    cfg = deldel_config or DelDelConfig(
        segments_target=18,
        near_frac=0.65,
        k_near_base=3,
        k_far_per_i=2,
        q_near=0.25,
        q_far=0.85,
        margin_quantile=0.2,
        secant_iters=1,
        final_bisect=3,
        min_logit_gain=0.0,
        min_pair_margin_end=0.0,
        prob_swing_weight=0.5,
        use_jsd=False,
        random_state=dataset_kwargs.get("random_state", 0),
    )

    engine = DelDel(cfg)
    t0 = perf_counter()
    engine.fit(X, model)
    stage_timings.append(
        dict(
            stage="deldel_fit",
            callable="deldel.engine.DelDel.fit",
            duration_s=perf_counter() - t0,
        )
    )
    records = list(engine.records_)

    frontier_kwargs = dict(frontier_kwargs or {})
    frontier_defaults = dict(
        min_cluster_size=3,
        max_models_per_round=6,
        max_depth=3,
        angle_merge_deg=8.5,
        offset_merge_tau=0.04,
        seed=dataset_kwargs.get("random_state", 0),
        prefer_cp=False,
    )
    frontier_defaults.update(frontier_kwargs)

    t0 = perf_counter()
    frontier = compute_frontier_planes_all_modes(records, **frontier_defaults)
    stage_timings.append(
        dict(
            stage="frontier_computation",
            callable="deldel.frontier_planes_all_modes.compute_frontier_planes_all_modes",
            duration_s=perf_counter() - t0,
        )
    )

    t0 = perf_counter()
    selection = _build_selection_from_frontier(frontier, X, y, feature_names)
    selection_source = "frontier"
    if not selection.get("by_pair_augmented"):
        selection = _build_demo_selection(X, y, feature_names)
        selection_source = "demo_fallback"
    selection_callable = (
        "deldel.experiments._build_selection_from_frontier"
        if selection_source == "frontier"
        else "deldel.experiments._build_demo_selection"
    )
    stage_timings.append(
        dict(
            stage="selection_build",
            callable=selection_callable,
            duration_s=perf_counter() - t0,
            selection_source=selection_source,
        )
    )

    finder_param_grid = list(finder_param_grid) if finder_param_grid is not None else [
        dict(
            max_planes_in_rule=1,
            max_planes_per_pair=1,
            min_support=15,
            min_rel_gain_f1=0.05,
            min_abs_gain_f1=0.02,
            min_lift_prec=1.05,
            min_abs_gain_prec=0.02,
            consider_dims_up_to=2,
            rng_seed=0,
        ),
        dict(
            max_planes_in_rule=2,
            max_planes_per_pair=2,
            min_support=20,
            min_rel_gain_f1=0.08,
            min_abs_gain_f1=0.03,
            min_lift_prec=1.10,
            min_abs_gain_prec=0.03,
            consider_dims_up_to=3,
            rng_seed=1,
        ),
        dict(
            max_planes_in_rule=3,
            max_planes_per_pair=3,
            min_support=25,
            min_rel_gain_f1=0.12,
            min_abs_gain_f1=0.04,
            min_lift_prec=1.15,
            min_abs_gain_prec=0.04,
            consider_dims_up_to=4,
            rng_seed=2,
            heuristic_merge_enable=True,
            p_merge=0.2,
        ),
    ]

    finder_runs: List[Dict[str, Any]] = []
    n_dims = X.shape[1]

    for idx, params in enumerate(finder_param_grid):
        params = dict(params)
        t_run = perf_counter()
        valuable = find_low_dim_spaces(
            X,
            y,
            selection,
            feature_names=list(feature_names),
            **params,
        )
        runtime = perf_counter() - t_run

        counts_by_dim = {dim: len(valuable.get(dim, [])) for dim in range(1, n_dims + 1)}
        best_f1 = {}
        best_precision = {}
        for dim, regions in valuable.items():
            if not regions:
                best_f1[dim] = 0.0
                best_precision[dim] = 0.0
                continue
            best_f1[dim] = max(float(r.get("metrics", {}).get("f1", 0.0)) for r in regions)
            best_precision[dim] = max(float(r.get("metrics", {}).get("precision", 0.0)) for r in regions)

        stage_timings.append(
            _stage_entry(
                stage=f"find_low_dim_spaces_run_{idx}",
                callable="deldel.find_low_dim_spaces_fast.find_low_dim_spaces",
                start_s=t_run,
                end_s=t_run + runtime,
                extra=dict(
                    run_index=idx,
                    params=json.dumps(params, sort_keys=True),
                    regions_total=sum(len(valuable.get(dim, [])) for dim in valuable),
                    best_f1=json.dumps(best_f1, sort_keys=True),
                    best_precision=json.dumps(best_precision, sort_keys=True),
                ),
            )
        )

        finder_runs.append(
            dict(
                run_index=idx,
                runtime_s=runtime,
                params=params,
                regions_total=sum(counts_by_dim.values()),
                regions_by_dim=counts_by_dim,
                best_f1_by_dim=best_f1,
                best_precision_by_dim=best_precision,
            )
        )

    csv_outputs: Dict[str, str] = {}
    if csv_dir is not None:
        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stages_path = out_dir / "stage_timings.csv"
        with stages_path.open("w", newline="") as fh:
            _write_stage_timings_csv(stage_timings, fh)
        csv_outputs["stage_timings"] = str(stages_path)

        finder_path = out_dir / "finder_runs.csv"
        fieldnames = [
            "run_index",
            "runtime_s",
            "params_json",
            "regions_total",
        ]
        for dim in range(1, n_dims + 1):
            fieldnames.append(f"regions_dim{dim}")
            fieldnames.append(f"best_f1_dim{dim}")
            fieldnames.append(f"best_precision_dim{dim}")

        with finder_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in finder_runs:
                row = dict(
                    run_index=entry["run_index"],
                    runtime_s=f"{entry['runtime_s']:.6f}",
                    params_json=json.dumps(entry["params"], sort_keys=True),
                    regions_total=entry["regions_total"],
                )
                for dim in range(1, n_dims + 1):
                    row[f"regions_dim{dim}"] = entry["regions_by_dim"].get(dim, 0)
                    row[f"best_f1_dim{dim}"] = f"{entry['best_f1_by_dim'].get(dim, 0.0):.6f}"
                    row[f"best_precision_dim{dim}"] = (
                        f"{entry['best_precision_by_dim'].get(dim, 0.0):.6f}"
                    )
                writer.writerow(row)
        csv_outputs["finder_runs"] = str(finder_path)

    summary = dict(
        dataset=dict(
            n_samples=int(X.shape[0]),
            n_dims=int(X.shape[1]),
            feature_names=list(feature_names),
            class_counts={int(cls): int((y == int(cls)).sum()) for cls in np.unique(y)},
        ),
        stage_timings=stage_timings,
        records_total=len(records),
        frontier_pairs=len(frontier),
        selection_source=selection_source,
        finder_runs=finder_runs,
        csv_outputs=csv_outputs,
    )
    return summary


def run_low_dim_spaces_demo(
    *,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    finder_kwargs: Optional[Mapping[str, Any]] = None,
    csv_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute :func:`find_low_dim_spaces` on a synthetic dataset.

    This example mirrors the interactive snippet distributed in the project
    documentation.  The helper builds a manageable dataset, synthesises a
    selection payload, discovers valuable low-dimensional regions, and writes a
    CSV file when ``csv_dir`` is provided.
    """

    dataset_kwargs = dict(dataset_kwargs or {})
    dataset_kwargs.setdefault("n_per_cluster", 120)
    dataset_kwargs.setdefault("std_class1", 0.45)
    dataset_kwargs.setdefault("std_other", 0.75)
    dataset_kwargs.setdefault("random_state", 0)

    X, y, feature_names = make_corner_class_dataset(**dataset_kwargs)
    sel = _build_demo_selection(X, y, feature_names)

    finder_defaults = dict(
        max_planes_in_rule=2,
        max_planes_per_pair=1,
        min_support=20,
        min_rel_gain_f1=0.05,
        min_abs_gain_f1=0.02,
        min_lift_prec=1.05,
        min_abs_gain_prec=0.02,
        consider_dims_up_to=2,
        rng_seed=0,
    )
    if finder_kwargs:
        finder_defaults.update(finder_kwargs)

    valuable = find_low_dim_spaces(
        X,
        y,
        sel,
        feature_names=list(feature_names),
        **finder_defaults,
    )

    csv_path: Optional[Path] = None
    if csv_dir is not None:
        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "low_dim_spaces.csv"
        with csv_path.open("w", newline="") as fh:
            fieldnames = [
                "dimensionality",
                "target_class",
                "region_id",
                "rule_text",
                "precision",
                "recall",
                "f1",
                "size",
                "lift_precision",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for dim in sorted(valuable.keys()):
                for rec in valuable[dim]:
                    metrics = rec["metrics"]
                    writer.writerow(
                        dict(
                            dimensionality=dim,
                            target_class=rec["target_class"],
                            region_id=rec["region_id"],
                            rule_text=rec.get("rule_text", ""),
                            precision=f"{metrics['precision']:.6f}",
                            recall=f"{metrics['recall']:.6f}",
                            f1=f"{metrics['f1']:.6f}",
                            size=int(metrics["size"]),
                            lift_precision=f"{metrics['lift_precision']:.6f}",
                        )
                    )

    return dict(
        X=X,
        y=y,
        feature_names=list(feature_names),
        selection=sel,
        valuable=valuable,
        csv_path=str(csv_path) if csv_path is not None else None,
    )
