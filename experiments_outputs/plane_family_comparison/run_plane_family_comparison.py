from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn import datasets


@dataclass
class DatasetSpec:
    name: str
    loader: callable


@dataclass
class RepoSpec:
    label: str
    path: Path


# -----------------------------
# Helpers
# -----------------------------


def _axis_aligned_frontier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    thresholds_per_dim: int = 2,
) -> Dict[Tuple[int, int], Dict[str, dict]]:
    """Generate a lightweight frontier payload built from axis-aligned cuts."""

    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    classes = sorted(np.unique(y))
    res: Dict[Tuple[int, int], Dict[str, dict]] = {}

    for a, b in ((int(i), int(j)) for i in classes for j in classes if i < j):
        mask_a = y == int(a)
        mask_b = y == int(b)
        Xa, Xb = X[mask_a], X[mask_b]
        planes_by_label: Dict[int, List[dict]] = {int(a): [], int(b): []}

        for dim in range(X.shape[1]):
            va = Xa[:, dim]
            vb = Xb[:, dim]
            if va.size == 0 or vb.size == 0:
                continue

            mean_a = float(np.mean(va))
            mean_b = float(np.mean(vb))
            joint = np.concatenate([va, vb])
            span = np.linspace(mean_a, mean_b, thresholds_per_dim + 2)[1:-1]
            percentiles = np.percentile(joint, [25, 50, 75])
            thresholds = sorted({float(t) for t in np.concatenate([span, percentiles])})

            for thr in thresholds:
                n = np.zeros(X.shape[1], dtype=float)
                n[dim] = 1.0
                plane = dict(n=n.tolist(), b=float(-thr))
                planes_by_label[int(a)].append(dict(plane))
                planes_by_label[int(b)].append(dict(plane))

        res[(int(a), int(b))] = dict(planes_by_label=planes_by_label)

    return res


def _load_prune_callable(repo: RepoSpec):
    """Import prune_and_orient_planes_unified_globalmaj from the given repo."""

    repo_src = repo.path / "src"
    sys.path.insert(0, str(repo_src))
    # Purge any existing deldel modules to avoid cross-version reuse.
    for name in [m for m in list(sys.modules) if m.startswith("deldel")]:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        from deldel.globalmaj import prune_and_orient_planes_unified_globalmaj  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive import
        raise RuntimeError(f"No se pudo importar prune_and_orient_planes_unified_globalmaj desde {repo.path}") from exc
    return prune_and_orient_planes_unified_globalmaj


def _best_metrics_for_plane(plane: Dict[str, any]) -> Dict[str, float]:
    metrics_pair = plane.get("metrics_pair", {}) or {}
    cls = plane.get("best_class_pair")
    if cls is None:
        cls = plane.get("best_class") if plane.get("best_class") is not None else plane.get("target_class")
    metrics = metrics_pair.get(int(cls), {}) if cls is not None else {}
    out = {}
    for key in ("precision", "recall", "f1", "lift"):
        if key in metrics and metrics[key] is not None:
            try:
                out[key] = float(metrics[key])
            except Exception:
                continue
    return out


def _aggregate_metrics(planes: Sequence[Dict[str, any]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    if not planes:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "lift": np.nan}
    metrics_by_key: Dict[str, List[float]] = {k: [] for k in ("precision", "recall", "f1", "lift")}
    for plane in planes:
        pm = _best_metrics_for_plane(plane)
        for k, v in pm.items():
            metrics_by_key[k].append(v)
    for k, vals in metrics_by_key.items():
        agg[k] = float(np.mean(vals)) if vals else np.nan
    return agg


def _planes_per_family_stats(planes: Sequence[Dict[str, any]]) -> Tuple[int, float, int]:
    if not planes:
        return 0, float("nan"), 0
    counter: Counter = Counter()
    for p in planes:
        fid = p.get("family_id")
        if fid is None:
            fid = p.get("plane_id")
        counter[str(fid)] += 1
    counts = list(counter.values())
    return len(counter), float(np.mean(counts)), int(max(counts))


# -----------------------------
# Core experiment
# -----------------------------


def run_experiment(repo: RepoSpec, datasets: Sequence[DatasetSpec], output_rows: List[Dict[str, any]]):
    prune = _load_prune_callable(repo)

    for ds in datasets:
        X, y, feature_names = ds.loader()
        res = _axis_aligned_frontier(X, y)

        start = perf_counter()
        selection = prune(
            res,
            X,
            y,
            feature_names=list(feature_names),
            min_abs_diff=0.01,
            min_rel_lift=0.05,
            min_region_size=10,
            max_k=6,
            min_recall=0.30,
            min_region_frac=0.02,
            diversity_additions=False,
            global_sides="good",
            refine_topK_per_class=150,
            family_clustering_mode="dbscan",
            family_dbscan_eps=0.35,
            verbosity=0,
        )
        runtime = perf_counter() - start

        for pair, payload in selection.get("by_pair_augmented", {}).items():
            winning = payload.get("winning_planes", []) or []
            others = payload.get("other_planes", []) or []
            pool = list(winning) + list(others)
            fams, mean_per_family, max_per_family = _planes_per_family_stats(pool)
            metrics_win = _aggregate_metrics(winning)
            metrics_oth = _aggregate_metrics(others)

            output_rows.append(
                dict(
                    repo_label=repo.label,
                    repo_path=str(repo.path),
                    dataset=ds.name,
                    n_samples=int(X.shape[0]),
                    n_features=int(X.shape[1]),
                    pair=f"{int(pair[0])}-{int(pair[1])}",
                    runtime_s=runtime,
                    winning_count=len(winning),
                    other_count=len(others),
                    total_pool=len(pool),
                    families=fams,
                    avg_planes_per_family=mean_per_family,
                    max_planes_per_family=max_per_family,
                    winners_precision=metrics_win.get("precision", np.nan),
                    winners_recall=metrics_win.get("recall", np.nan),
                    winners_f1=metrics_win.get("f1", np.nan),
                    winners_lift=metrics_win.get("lift", np.nan),
                    others_precision=metrics_oth.get("precision", np.nan),
                    others_recall=metrics_oth.get("recall", np.nan),
                    others_f1=metrics_oth.get("f1", np.nan),
                    others_lift=metrics_oth.get("lift", np.nan),
                )
            )


# -----------------------------
# Datasets
# -----------------------------


def _load_iris():
    data = datasets.load_iris()
    return data.data, data.target, data.feature_names


def _load_wine():
    data = datasets.load_wine()
    return data.data, data.target, data.feature_names


def _load_synthetic():
    X, y = datasets.make_classification(
        n_samples=800,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=4,
        n_clusters_per_class=2,
        class_sep=1.4,
        random_state=7,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, feature_names


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Comparación de versiones para prune_and_orient_planes_unified_globalmaj")
    parser.add_argument(
        "--version",
        action="append",
        dest="versions",
        help="Especifica una versión como etiqueta=path (ej. actual=.). Se puede repetir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("plane_family_comparison.csv"),
        help="Ruta del CSV de salida",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_specs: List[RepoSpec] = []

    if args.versions:
        for item in args.versions:
            if "=" not in item:
                raise SystemExit("Cada --version debe tener el formato etiqueta=path")
            label, path_str = item.split("=", 1)
            repo_specs.append(RepoSpec(label=label, path=Path(path_str).resolve()))
    else:
        repo_specs.append(RepoSpec(label="current", path=Path(__file__).resolve().parents[2]))
        prev_path = Path(__file__).resolve().parents[2] / ".." / "PureSheShe_prev"
        repo_specs.append(RepoSpec(label="previous", path=prev_path.resolve()))

    datasets_spec = [
        DatasetSpec("iris", _load_iris),
        DatasetSpec("wine", _load_wine),
        DatasetSpec("synthetic_mix", _load_synthetic),
    ]

    output_rows: List[Dict[str, any]] = []

    for repo in repo_specs:
        if not repo.path.exists():
            raise SystemExit(f"No existe la ruta para la versión {repo.label}: {repo.path}")
        run_experiment(repo, datasets_spec, output_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "repo_label",
                "repo_path",
                "dataset",
                "n_samples",
                "n_features",
                "pair",
                "runtime_s",
                "winning_count",
                "other_count",
                "total_pool",
                "families",
                "avg_planes_per_family",
                "max_planes_per_family",
                "winners_precision",
                "winners_recall",
                "winners_f1",
                "winners_lift",
                "others_precision",
                "others_recall",
                "others_f1",
                "others_lift",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"CSV guardado en {args.output}")


if __name__ == "__main__":
    main()
