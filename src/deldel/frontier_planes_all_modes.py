from __future__ import annotations

from typing import Iterable, Tuple, List, Dict, Optional, Any, Sequence

import logging
from time import perf_counter

import numpy as np

from deldel.utils import _verbosity_to_level
from .engine import DeltaRecord


# ===================== compute_frontier_planes_all_modes — Modo C (mejorado) =====================
# -----------------------------------------------------------------------------------------------
# Utilidades generales
# -----------------------------------------------------------------------------------------------

def _unit(v: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=axis, keepdims=keepdims) + 1e-12
    return v / n


def _endpoints(r: DeltaRecord, prefer_cp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (a, b) para un ``DeltaRecord``."""

    a = np.asarray(r.x0, float)
    if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x", None)).size > 0:
        b = np.asarray(r.cp_x, float).reshape(-1, a.size).mean(axis=0)
    else:
        b = np.asarray(r.x1, float)
    return a, b


def _segment_arrays(
    records: Iterable[DeltaRecord],
    pair: Tuple[int, int],
    prefer_cp: bool = True,
    *,
    canonicalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construye arreglos por par de clases ``{a, b}``."""

    A, B, U, M, L, W, idx_map = [], [], [], [], [], [], []
    a_cls, b_cls = map(int, pair)
    for idx, r in enumerate(records):
        if not getattr(r, "success", True):
            continue
        if {int(r.y0), int(r.y1)} != {a_cls, b_cls}:
            continue
        a, b = _endpoints(r, prefer_cp)
        v = b - a
        lenv = float(np.linalg.norm(v))
        if lenv <= 1e-12:
            continue
        u = v / lenv
        if canonicalize and u[0] < 0:
            u = -u
        A.append(a)
        B.append(b)
        U.append(u)
        M.append(0.5 * (a + b))
        L.append(lenv)
        W.append(float(getattr(r, "final_score", 1.0)))
        idx_map.append(idx)
    if not A:
        raise ValueError("No hay segmentos para ese par.")
    return (
        np.vstack(A),
        np.vstack(B),
        np.vstack(U),
        np.vstack(M),
        np.asarray(L, float),
        np.asarray(W, float),
        np.asarray(idx_map, int),
    )


# -----------------------------------------------------------------------------------------------
# Ajuste robusto de planos (TLS + LO + LTS + IRLS SAFE)
# -----------------------------------------------------------------------------------------------

def _fit_plane_tls(P: np.ndarray) -> Tuple[np.ndarray, float]:
    P = np.asarray(P, float)
    mu = P.mean(axis=0, keepdims=True)
    Z = P - mu
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    b = -float(n @ mu.reshape(-1))
    return n, b


def _residuals(P: np.ndarray, n: np.ndarray, b: float) -> np.ndarray:
    return (P @ n.reshape(-1)) + float(b)


def _weights_robust(r: np.ndarray, loss: str = "huber", c: float = 1.345) -> np.ndarray:
    r = np.asarray(r, float).reshape(-1)
    if loss == "huber":
        ar = np.abs(r)
        w = np.where(ar <= c, 1.0, c / (ar + 1e-12))
    else:  # Tukey biweight
        u = np.clip(r / (c + 1e-12), -1.0, 1.0)
        w = (1.0 - u**2) ** 2
    return w.astype(float)


def _refine_plane_irls(
    P: np.ndarray,
    n: np.ndarray,
    b: float,
    iters: int = 15,
    loss: str = "huber",
    c: float = 1.345,
) -> Tuple[np.ndarray, float]:
    """Refina un plano mediante IRLS robusto."""

    P = np.asarray(P, float)
    nn = np.asarray(n, float).reshape(-1).copy()
    bb = float(b)
    for _ in range(int(iters)):
        r = _residuals(P, nn, bb).reshape(-1)
        w = _weights_robust(r, loss=loss, c=c).reshape(-1)
        sw = float(w.sum()) + 1e-12
        muw = (w[:, None] * P).sum(axis=0) / sw
        Z = P - muw[None, :]
        Cw = (Z.T * w) @ Z / sw
        evals, evecs = np.linalg.eigh(Cw)
        nn = evecs[:, int(np.argmin(evals))]
        nn = nn / (np.linalg.norm(nn) + 1e-12)
        bb = -float(nn @ muw)
    return nn, float(bb)


def _stats(P: np.ndarray, n: np.ndarray, b: float) -> Dict[str, float]:
    r = np.abs(_residuals(P, n, b)).reshape(-1)
    mad = np.median(np.abs(r - np.median(r)))
    return dict(
        mean=float(np.mean(r)),
        median=float(np.median(r)),
        mad=float(mad),
        rmse=float(np.sqrt(np.mean(r**2))),
    )


def _planarity(P: np.ndarray) -> float:
    if P.shape[0] < 2:
        return 0.0
    Z = P - P.mean(axis=0, keepdims=True)
    C = (Z.T @ Z) / max(1, P.shape[0] - 1)
    evs = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    s = float(evs.sum()) + 1e-12
    return float(1.0 - (float(evs.min()) / s))


def _lts_trim(
    P: np.ndarray,
    n: np.ndarray,
    b: float,
    trim_frac: float = 0.2,
    min_keep: Optional[int] = None,
) -> np.ndarray:
    r = np.abs(_residuals(P, n, b)).reshape(-1)
    order = np.argsort(r)
    keep = int(np.ceil((1.0 - float(trim_frac)) * P.shape[0]))
    if min_keep is not None:
        keep = max(int(min_keep), keep)
    idx = order[:keep]
    return idx


def _multi_ransac_lo_lts(
    P: np.ndarray,
    max_models: int = 4,
    max_iter: int = 600,
    tau: Optional[float] = None,
    seed: int = 0,
    trim_frac: float = 0.2,
    dplus1: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """RANSAC multi-modelos + LO + LTS + IRLS para planos."""

    rng = np.random.RandomState(seed)
    d = int(P.shape[1])
    s = d + 1 if dplus1 is None else int(dplus1)
    remain = np.arange(P.shape[0])
    cands: List[Dict[str, Any]] = []

    if tau is None:
        n0, b0 = _fit_plane_tls(P)
        r0 = np.abs(_residuals(P, n0, b0)).reshape(-1)
        mad = float(np.median(np.abs(r0 - np.median(r0)))) + 1e-12
        tau = 2.5 * mad if mad > 0 else (1.5 * float(np.std(r0)) + 1e-9)

    for _ in range(int(max_models)):
        if remain.size < s:
            break
        best_cnt, best_n, best_b, best_mask = -1, None, None, None
        for _ in range(int(max_iter)):
            idx = rng.choice(remain, size=s, replace=False)
            try:
                n, b = _fit_plane_tls(P[idx])
            except Exception:
                continue
            r = np.abs(_residuals(P, n, b)).reshape(-1)
            mask = r <= float(tau)
            cnt = int(mask.sum())
            if cnt > best_cnt:
                best_cnt, best_n, best_b, best_mask = cnt, n, b, mask

        if best_cnt < s:
            break

        n, b = _fit_plane_tls(P[best_mask])
        for _ in range(2):
            r_in = np.abs(_residuals(P, n, b)).reshape(-1)
            mad = float(np.median(np.abs(r_in[best_mask] - np.median(r_in[best_mask])))) + 1e-12
            tau_lo = max(float(tau), 2.5 * mad)
            best_mask = r_in <= tau_lo
            if int(best_mask.sum()) < s:
                break
            n, b = _fit_plane_tls(P[best_mask])

        it_lts = 0
        while True:
            it_lts += 1
            r_all = np.abs(_residuals(P, n, b)).reshape(-1)
            med = float(np.median(r_all[best_mask])) if best_mask.any() else float(np.median(r_all))
            if med <= float(tau) or it_lts > 3 or int(best_mask.sum()) < s:
                break
            keep_idx = _lts_trim(P[best_mask], n, b, trim_frac=trim_frac, min_keep=s)
            mask_keep = np.zeros_like(best_mask)
            mask_idx = np.flatnonzero(best_mask)[keep_idx]
            mask_keep[mask_idx] = True
            best_mask = mask_keep
            n, b = _fit_plane_tls(P[best_mask])

        n, b = _refine_plane_irls(P[best_mask], n, b, iters=12, loss="huber", c=1.345)

        cands.append(
            {
                "n": n,
                "b": float(b),
                "inliers": np.flatnonzero(best_mask),
                "tau": float(tau),
            }
        )
        remain = np.setdiff1d(remain, np.flatnonzero(best_mask), assume_unique=False)
        if remain.size < s:
            break

    return cands, float(tau)


def _merge_near_parallel(
    planes: List[Dict[str, Any]],
    P: np.ndarray,
    angle_deg: float = 6.0,
    offset_tol: float = 0.02,
) -> List[Dict[str, Any]]:
    if len(planes) <= 1:
        return planes
    out: List[Dict[str, Any]] = []
    used = np.zeros(len(planes), bool)
    for i in range(len(planes)):
        if used[i]:
            continue
        ni, bi = planes[i]["n"], planes[i]["b"]
        group = [i]
        used[i] = True
        for j in range(i + 1, len(planes)):
            if used[j]:
                continue
            nj, bj = planes[j]["n"], planes[j]["b"]
            ang = np.degrees(np.arccos(np.clip(np.abs(float(ni @ nj)), -1.0, 1.0)))
            if ang <= angle_deg and abs(bi - bj) <= offset_tol:
                group.append(j)
                used[j] = True
        if len(group) == 1:
            out.append(planes[i])
        else:
            idxs = np.unique(np.concatenate([planes[g]["inliers"] for g in group]))
            n, b = _fit_plane_tls(P[idxs])
            n, b = _refine_plane_irls(P[idxs], n, b, iters=10, loss="huber", c=1.345)
            out.append(
                {
                    "n": n,
                    "b": float(b),
                    "inliers": idxs,
                    "tau": np.median([planes[g]["tau"] for g in group]),
                }
            )
    return out


def _score_plane_full(P: np.ndarray, n: np.ndarray, b: float, tau: float) -> Dict[str, float]:
    r = np.abs(_residuals(P, n, b)).reshape(-1)
    inliers = r <= float(tau)
    plan = _planarity(P[inliers]) if inliers.any() else 0.0

    def _stats_dict(Q: np.ndarray) -> Dict[str, float]:
        if Q.shape[0] == 0:
            return dict(mean=np.inf, rmse=np.inf)
        rr = np.abs(_residuals(Q, n, b)).reshape(-1)
        return dict(mean=float(np.mean(rr)), rmse=float(np.sqrt(np.mean(rr**2))))

    stg = _stats_dict(P)
    sti = _stats_dict(P[inliers] if inliers.any() else P)
    score = 1.0 * float(inliers.sum()) + 0.35 * plan - 0.35 * sti["mean"]
    return dict(
        score=float(score),
        coverage=float(inliers.sum()),
        planarity=plan,
        **stg,
        inlier_mean=sti["mean"],
        inlier_rmse=sti["rmse"],
    )


def _plane_report(
    P: np.ndarray,
    plane: Dict[str, Any],
    label_id: int,
    dims: Optional[Tuple[int, ...]] = None,
    dim_names: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    n, b = plane["n"], plane["b"]
    inl = plane["inliers"]
    tau = float(plane.get("tau", 0.0))
    mu_inl = P[inl].mean(axis=0) if len(inl) > 0 else P.mean(axis=0)
    stat_g = _stats(P, n, b)
    stat_i = _stats(P[inl], n, b) if len(inl) > 0 else stat_g
    dims = tuple(range(P.shape[1])) if dims is None else tuple(int(i) for i in dims)
    dim_names = tuple(dim_names) if dim_names is not None else None
    rep = dict(
        n=n,
        b=float(b),
        mu=mu_inl,
        count=int(len(inl)),
        label=int(label_id),
        tau=tau,
        dims=dims,
        dim_names=dim_names,
        fit_error=dict(
            global_mean=stat_g["mean"],
            global_rmse=stat_g["rmse"],
            inlier_mean=stat_i["mean"],
            inlier_rmse=stat_i["rmse"],
        ),
    )
    rep.update(_score_plane_full(P, n, b, tau))
    return rep


# -----------------------------------------------------------------------------------------------
# Perpendicularidad / intersección / bimodalidad (para C)
# -----------------------------------------------------------------------------------------------

def _ray_plane_params(a: np.ndarray, u: np.ndarray, n: np.ndarray, b: float) -> Tuple[float, float]:
    den = float(n @ u)
    if abs(den) < 1e-12:
        return np.inf, 0.0
    t = -float(n @ a + b) / den
    cos_inc = abs(den) / (np.linalg.norm(n) * np.linalg.norm(u) + 1e-12)
    return t, cos_inc


def _maybe_split_bimodal_by_sign(
    U: np.ndarray,
    n: np.ndarray,
    *,
    min_side_frac: float = 0.25,
    mean_abs_max: float = 0.2,
) -> Optional[np.ndarray]:
    if U.size == 0:
        return None
    proj = U @ n.reshape(-1)
    frac_pos = float((proj >= 0).mean())
    frac_neg = 1.0 - frac_pos
    mean_abs = float(np.mean(np.abs(proj)))
    if frac_pos >= min_side_frac and frac_neg >= min_side_frac and mean_abs <= mean_abs_max:
        return proj >= 0
    return None


# -----------------------------------------------------------------------------------------------
# Clustering C (VMF/Offsets) con perpendicularidad, t>0 y split bimodal
# -----------------------------------------------------------------------------------------------

def _cluster_vmf_offsets_perp(
    U: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    *,
    Kmax: int = 12,
    random_state: int = 0,
    s_min: float = 0.80,
    t_min: float = 1e-6,
    min_bucket: int = 8,
) -> np.ndarray:
    from sklearn.cluster import DBSCAN, KMeans, OPTICS
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    n_total, d = U.shape

    bestK, best, bestlab, bestcent = 1, -1e9, None, None
    rng = np.random.RandomState(random_state)
    sub = rng.choice(U.shape[0], size=min(1500, U.shape[0]), replace=False)
    for K in range(1, max(1, Kmax) + 1):
        km = KMeans(n_clusters=K, n_init=10, random_state=random_state).fit(U)
        lab = km.labels_
        s = silhouette_score(U[sub], lab[sub], metric="cosine") if K > 1 else -0.1
        score = s - 0.05 * K
        if score > best:
            best, bestK, bestlab, bestcent = score, K, lab, km.cluster_centers_
    labels_or, centers = bestlab, bestcent

    # Revisión angular para dividir clusters muy dispersos antes de pasar a offsets
    labels_refined = labels_or.copy()
    centers_map: Dict[int, np.ndarray] = {
        int(k): centers[int(k)] / (np.linalg.norm(centers[int(k)]) + 1e-12)
        for k in np.unique(labels_or)
    }
    next_label = int(labels_refined.max()) + 1
    to_check = list(np.unique(labels_refined))
    min_cos_threshold = 0.55
    spread_threshold = 0.25
    min_split_size = max(8, 2 * d)

    while to_check:
        k = int(to_check.pop(0))
        idx = np.flatnonzero(labels_refined == k)
        if idx.size < min_split_size:
            continue

        dmean = centers_map[k] / (np.linalg.norm(centers_map[k]) + 1e-12)
        cos_vals = U[idx] @ dmean
        min_cos = float(cos_vals.min())
        spread = float(cos_vals.max() - cos_vals.min())
        if min_cos <= min_cos_threshold and spread >= spread_threshold:
            km_loc = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit(U[idx])
            lab_loc = km_loc.labels_
            labels_refined[idx[lab_loc == 0]] = k
            labels_refined[idx[lab_loc == 1]] = next_label
            centers_map[k] = km_loc.cluster_centers_[0] / (
                np.linalg.norm(km_loc.cluster_centers_[0]) + 1e-12
            )
            centers_map[next_label] = km_loc.cluster_centers_[1] / (
                np.linalg.norm(km_loc.cluster_centers_[1]) + 1e-12
            )
            to_check.append(next_label)
            next_label += 1

    uniq = np.unique(labels_refined)
    relabel = {int(old): i for i, old in enumerate(uniq)}
    labels_or = np.array([relabel[int(l)] for l in labels_refined], int)
    centers = np.vstack([centers_map[int(k)] for k in uniq])

    all_labels = -np.ones(U.shape[0], int)
    next_base = 0

    for k in np.unique(labels_or):
        idx = np.flatnonzero(labels_or == k)
        if idx.size == 0:
            continue

        dmean = centers[int(k)]
        dmean = dmean / (np.linalg.norm(dmean) + 1e-12)
        P = np.eye(d) - np.outer(dmean, dmean)
        Xoff = M[idx] @ P.T

        n_bucket = int(idx.size)
        if n_bucket < (d + 2):
            local_lab = np.zeros(n_bucket, int)
        else:
            base_min_samples = max(5, int(np.ceil(np.log2(n_bucket))) + 1)
            base_min_cluster_size = max(6, 2 * d)
            min_samples = max(2, min(base_min_samples, n_bucket))
            min_cluster_size = max(2, min(base_min_cluster_size, n_bucket))
            optics = OPTICS(
                min_samples=min_samples,
                xi=0.05,
                min_cluster_size=min_cluster_size,
                n_jobs=None,
            )
            optics.fit(Xoff)
            local_lab = optics.labels_

            if (local_lab == -1).all():
                nloc = n_bucket
                k0 = max(2, min(int(np.ceil(np.log2(nloc))) + 1, nloc - 1))
                try:
                    nn = NearestNeighbors(n_neighbors=k0).fit(Xoff)
                    dists, _ = nn.kneighbors(Xoff)
                    kth = np.sort(dists[:, -1])
                    q = 0.6 if nloc >= 15 else (0.7 if nloc >= 8 else 0.8)
                    eps = float(np.quantile(kth, q))
                    local_lab = DBSCAN(
                        eps=max(eps, 1e-12),
                        min_samples=max(2, min(k0, nloc)),
                    ).fit_predict(Xoff)
                except Exception:
                    local_lab = np.full(nloc, -1, int)

                if (local_lab == -1).all():
                    if nloc < 6:
                        local_lab = np.zeros(nloc, int)
                    else:
                        z = PCA(n_components=1, random_state=random_state).fit_transform(Xoff).reshape(-1)
                        km = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit(
                            z.reshape(-1, 1)
                        )
                        local_lab = km.labels_

        for lab_val in np.unique(local_lab):
            I = idx[local_lab == lab_val]
            if I.size == 0:
                continue

            n0, b0 = _fit_plane_tls(F[I])
            n0, b0 = _refine_plane_irls(F[I], n0, b0, iters=8, loss="huber", c=1.345)

            keep = np.zeros(I.size, bool)
            for j, ii in enumerate(I):
                t, s = _ray_plane_params(M[ii], U[ii], n0, b0)
                keep[j] = (t > t_min) and (s >= s_min)

            I2 = I[keep]
            if I2.size < min(min_bucket, I.size // 3):
                I2 = I

            split_mask = _maybe_split_bimodal_by_sign(U[I2], n0, min_side_frac=0.25, mean_abs_max=0.2)
            if split_mask is None:
                all_labels[I2] = next_base
                next_base += 1
            else:
                I_pos = I2[split_mask]
                I_neg = I2[~split_mask]
                if I_pos.size:
                    all_labels[I_pos] = next_base
                    next_base += 1
                if I_neg.size:
                    all_labels[I_neg] = next_base
                    next_base += 1

    mask_un = all_labels < 0
    if mask_un.any():
        m = int(mask_un.sum())
        all_labels[mask_un] = np.arange(next_base, next_base + m)
    return all_labels


# -----------------------------------------------------------------------------------------------
# Pipeline de ajuste por cluster (igual lógica; opera sobre P = F)
# -----------------------------------------------------------------------------------------------

def _refine_divide_reassign_cluster(
    P: np.ndarray,
    label_id: int,
    max_models: int = 4,
    angle_merge_deg: float = 6.0,
    offset_merge_tau: float = 0.02,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    d = int(P.shape[1])
    s = d + 1
    cands, tau = _multi_ransac_lo_lts(P, max_models=max_models, max_iter=600, seed=seed, dplus1=s)
    if not cands:
        return [], -np.ones(P.shape[0], int), np.full(P.shape[0], np.inf)
    cands = _merge_near_parallel(cands, P, angle_deg=angle_merge_deg, offset_tol=offset_merge_tau)
    planes_label: List[Dict[str, Any]] = []
    for c in cands:
        c["tau"] = float(tau) if "tau" not in c else float(c["tau"])
        planes_label.append(c)
    H = np.column_stack(
        [np.abs(_residuals(P, pl["n"], float(pl["b"]))).reshape(-1) for pl in planes_label]
    )
    assign_idx = np.argmin(H, axis=1).astype(int)
    residuals = H[np.arange(P.shape[0]), assign_idx]
    return planes_label, assign_idx, residuals


def _recluster_discards(
    P: np.ndarray,
    mask_assigned: np.ndarray,
    mode: str,
    A: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    min_cluster_size: int = 8,
    seed: int = 0,
) -> np.ndarray:
    idx_disc = np.flatnonzero(~mask_assigned)
    sub_lab = -np.ones(P.shape[0], int)
    if idx_disc.size < max(3, min_cluster_size):
        return sub_lab
    if mode == "C":
        lab = _cluster_vmf_offsets_perp(
            U[idx_disc],
            M[idx_disc],
            F[idx_disc],
            Kmax=12,
            random_state=seed,
            s_min=0.80,
            t_min=1e-6,
        )
    else:
        raise ValueError("Solo se soporta modo 'C' en esta versión.")
    ref = -np.ones_like(lab)
    vals = np.unique(lab)
    for i, v in enumerate(vals):
        ref[lab == v] = i
    sub_lab[idx_disc] = ref
    return sub_lab


def _process_cluster(
    P: np.ndarray,
    label_id: int,
    mode: str,
    A: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    min_cluster_size: int = 8,
    seed: int = 0,
    max_depth: int = 2,
    max_models_per_round: int = 4,
    angle_merge_deg: float = 6.0,
    offset_merge_tau: float = 0.02,
    dims: Optional[Tuple[int, ...]] = None,
    dim_names: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    planes, asg, res = _refine_divide_reassign_cluster(
        P,
        label_id,
        max_models=max_models_per_round,
        angle_merge_deg=angle_merge_deg,
        offset_merge_tau=offset_merge_tau,
        seed=seed,
    )
    if not planes:
        return [], asg, res

    reports = [_plane_report(P, pl, label_id, dims=dims, dim_names=dim_names) for pl in planes]
    mask_assigned = np.zeros(P.shape[0], bool)
    for pl in planes:
        mask_assigned[pl["inliers"]] = True

    depth = 0
    while depth < max_depth:
        depth += 1
        sub_lab = _recluster_discards(
            P,
            mask_assigned,
            mode,
            A,
            B,
            U,
            M,
            F,
            min_cluster_size=min_cluster_size,
            seed=seed + depth,
        )
        uvals = np.unique(sub_lab[sub_lab >= 0])
        if uvals.size == 0:
            break
        for v in uvals:
            disc_idx = np.flatnonzero(sub_lab == v)
            if disc_idx.size < (P.shape[1] + 1):
                continue
            P_sub = P[disc_idx]
            pl2, _, _ = _refine_divide_reassign_cluster(
                P_sub,
                label_id,
                max_models=max_models_per_round,
                angle_merge_deg=angle_merge_deg,
                offset_merge_tau=offset_merge_tau,
                seed=seed + depth,
            )
            if not pl2:
                continue
            for pl in pl2:
                pl["inliers"] = disc_idx[pl["inliers"]]
                reports.append(
                    _plane_report(
                        P,
                        pl,
                        label_id,
                        dims=dims,
                        dim_names=dim_names,
                    )
                )
                mask_assigned[pl["inliers"]] = True
                planes.append(pl)

    if planes:
        H = np.column_stack(
            [np.abs(_residuals(P, pl["n"], float(pl["b"]))).reshape(-1) for pl in planes]
        )
        asg_final = np.argmin(H, axis=1).astype(int)
        res_final = H[np.arange(P.shape[0]), asg_final]
    else:
        asg_final = asg
        res_final = res

    return reports, asg_final, res_final


def _extract_dims_from_explorer(
    explorer_reports: Optional[Sequence[Any]],
    *,
    feature_names: Optional[Sequence[str]],
    default_dim: int,
    top_k: int,
) -> List[Tuple[int, ...]]:
    """Convierte reportes/iterables de explorer en tuplas de índices de columna."""

    if not explorer_reports:
        return []

    name_to_idx: Dict[Any, int] = {}
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(default_dim)]
    for idx, name in enumerate(feature_names):
        name_to_idx[name] = idx

    dims_set: List[Tuple[int, ...]] = []
    for rep in list(explorer_reports)[:top_k]:
        if hasattr(rep, "features"):
            feats = getattr(rep, "features")
        else:
            feats = rep

        dims: List[int] = []
        for f in feats:
            if isinstance(f, (int, np.integer)):
                dims.append(int(f))
                continue
            if f in name_to_idx:
                dims.append(int(name_to_idx[f]))
                continue
            if isinstance(f, str) and f.startswith("x"):
                try:
                    dims.append(int(f[1:]))
                except ValueError:
                    continue
        dims = sorted(set(dims))
        if len(dims) >= 2:
            dims_set.append(tuple(dims))

    # de-dup preservando orden
    seen: set = set()
    uniq: List[Tuple[int, ...]] = []
    for dims in dims_set:
        if dims in seen:
            continue
        seen.add(dims)
        uniq.append(dims)
    return uniq


# -----------------------------------------------------------------------------------------------
# Ajuste por par con Modo C únicamente
# -----------------------------------------------------------------------------------------------

def _fit_for_pair_all(
    records: Iterable[DeltaRecord],
    pair: Tuple[int, int],
    mode: str = "C",
    prefer_cp: bool = True,
    min_cluster_size: int = 8,
    seed: int = 0,
    max_models_per_round: int = 4,
    max_depth: int = 2,
    angle_merge_deg: float = 6.0,
    offset_merge_tau: float = 0.02,
    dims: Optional[Sequence[int]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    A, B, U, M, L, W, rec_idx = _segment_arrays(
        records,
        pair,
        prefer_cp=prefer_cp,
        canonicalize=False,
    )
    _ = L, W  # suprime "unused" en modo C
    if dims is not None:
        dims = tuple(int(i) for i in dims)
        A = A[:, dims]
        B = B[:, dims]
        U = U[:, dims]
        M = M[:, dims]
    d = A.shape[1]
    eps = 1e-3
    F = B + eps * U

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]
    fnames = list(feature_names)
    dim_names = None
    if dims is not None:
        try:
            dim_names = tuple(fnames[i] for i in dims)
        except Exception:
            dim_names = None
    elif fnames:
        dim_names = tuple(fnames[:d])

    mode = str(mode).upper()
    if mode not in ("C", "VMF", "OFFSETS"):
        raise ValueError("Esta versión solo soporta mode in {'C','VMF','OFFSETS'}")

    labels = _cluster_vmf_offsets_perp(U, M, F, Kmax=8, random_state=seed, s_min=0.80, t_min=1e-6)
    mode_key = "C"

    planes_by_label: Dict[int, List[Dict[str, Any]]] = {}
    assigned_label = -np.ones(F.shape[0], int)
    assigned_plane = -np.ones(F.shape[0], int)
    residual_min = np.full(F.shape[0], np.inf)

    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if idx.size < (d + 1):
            continue
        Pk = F[idx]
        reports, asg_local, _ = _process_cluster(
            Pk,
            int(lab),
            mode_key,
            A[idx],
            B[idx],
            U[idx],
            M[idx],
            F[idx],
            min_cluster_size=min_cluster_size,
            seed=seed,
            max_depth=max_depth,
            max_models_per_round=max_models_per_round,
            angle_merge_deg=angle_merge_deg,
            offset_merge_tau=offset_merge_tau,
            dims=dims,
            dim_names=dim_names,
        )
        planes_by_label[int(lab)] = reports

        if reports:
            H = np.column_stack(
                [np.abs(_residuals(Pk, r["n"], float(r["b"]))).reshape(-1) for r in reports]
            )
            local_plane = np.argmin(H, axis=1).astype(int)
            assigned_label[idx] = int(lab)
            assigned_plane[idx] = local_plane
            residual_min[idx] = H[np.arange(Pk.shape[0]), local_plane]

    meta = dict(
        n_segments=int(F.shape[0]),
        n_labels_init=int(np.unique(labels).size),
        n_planes_total=int(sum(len(v) for v in planes_by_label.values())),
        dimension=int(d),
        dims=None if dims is None else tuple(int(i) for i in dims),
        dim_names=dim_names,
        mode=mode_key,
    )
    assignment = dict(
        rec_indices=rec_idx,
        assigned_label=assigned_label,
        assigned_plane=assigned_plane,
        residual=residual_min,
    )
    return dict(
        planes_by_label=planes_by_label,
        labels_cluster_init=labels,
        assignment=assignment,
        meta=meta,
    )


# -----------------------------------------------------------------------------------------------
# API principal — SOLO Modo C
# -----------------------------------------------------------------------------------------------

def compute_frontier_planes_all_modes(
    records: Iterable[DeltaRecord],
    pairs: Optional[Iterable[Tuple[int, int]]] = None,
    *,
    mode: str = "C",
    prefer_cp: bool = True,
    min_cluster_size: int = 8,
    seed: int = 0,
    max_models_per_round: int = 4,
    max_depth: int = 2,
    angle_merge_deg: float = 6.0,
    offset_merge_tau: float = 0.02,
    explorer_reports: Optional[Sequence[Any]] = None,
    explorer_feature_names: Optional[Sequence[str]] = None,
    explorer_top_k: int = 5,
    verbosity: Optional[int] = None,
    verbose: bool = False,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Encuentra planos frontera por par de clases usando exclusivamente el modo C mejorado."""

    logger = logging.getLogger(__name__)
    verbosity = 1 if (verbosity is None and verbose) else (-1 if verbosity is None else int(verbosity))
    level = _verbosity_to_level(verbosity)
    logger.setLevel(level)
    explorer_top_k = int(explorer_top_k)

    if pairs is None:
        seen = set()
        for r in records:
            if not getattr(r, "success", True):
                continue
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            seen.add(tuple(sorted((a, b))))
        pairs = sorted(seen)

    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    logger.log(
        level,
        "Inicio compute_frontier_planes_all_modes | mode=%s prefer_cp=%s min_cluster_size=%d total_pairs=%d",
        mode,
        prefer_cp,
        min_cluster_size,
        len(pairs),
    )
    for pair in pairs:
        pair_start = perf_counter()
        logger.log(level, "Procesando par %s", pair)
        try:
            base_block = _fit_for_pair_all(
                records,
                pair,
                mode=mode,
                prefer_cp=prefer_cp,
                min_cluster_size=min_cluster_size,
                seed=seed,
                max_models_per_round=max_models_per_round,
                max_depth=max_depth,
                angle_merge_deg=angle_merge_deg,
                offset_merge_tau=offset_merge_tau,
                feature_names=explorer_feature_names,
            )
        except Exception as exc:  # pragma: no cover - mantiene robustez de API
            dim_guess = None
            dim_names_guess: Optional[Tuple[str, ...]] = None
            try:
                first = next(iter(records))
                dim_guess = int(np.asarray(getattr(first, "x0", [])).reshape(-1).size)
                if explorer_feature_names:
                    dim_names_guess = tuple(explorer_feature_names[:dim_guess])
            except Exception:
                dim_guess = None
                dim_names_guess = None

            out[pair] = dict(
                error=str(exc),
                planes_by_label={},
                labels_cluster_init=np.array([], int),
                assignment=dict(
                    rec_indices=np.array([], int),
                    assigned_label=np.array([], int),
                    assigned_plane=np.array([], int),
                    residual=np.array([], float),
                ),
                meta=dict(
                    mode=mode,
                    dimension=dim_guess,
                    dims=None if dim_guess is None else tuple(range(dim_guess)),
                    dim_names=dim_names_guess,
                    subspace_error=str(exc),
                ),
            )
            logger.exception("Error al ajustar base para par %s", pair)
            continue

        try:
            base_dim = int(base_block.get("meta", {}).get("dimension") or 0)
            dims_list = _extract_dims_from_explorer(
                explorer_reports,
                feature_names=explorer_feature_names,
                default_dim=base_dim,
                top_k=explorer_top_k,
            )

            subspace_blocks: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for dims in dims_list:
                dims_start = perf_counter()
                logger.log(level, "Ajustando subespacio %s para par %s", dims, pair)
                subspace_blocks[dims] = _fit_for_pair_all(
                    records,
                    pair,
                    mode=mode,
                    prefer_cp=prefer_cp,
                    min_cluster_size=min_cluster_size,
                    seed=seed,
                    max_models_per_round=max_models_per_round,
                    max_depth=max_depth,
                    angle_merge_deg=angle_merge_deg,
                    offset_merge_tau=offset_merge_tau,
                    dims=dims,
                    feature_names=explorer_feature_names,
                )
                logger.log(
                    level,
                    "Subespacio %s completado en %.6f s",
                    dims,
                    perf_counter() - dims_start,
                )

            if subspace_blocks:
                base_block["subspace_variants"] = subspace_blocks
                base_block.setdefault("meta", {})["subspace_dims"] = list(subspace_blocks.keys())
                base_block["meta"]["n_planes_subspaces"] = int(
                    sum(
                        sum(len(v) for v in blk.get("planes_by_label", {}).values())
                        for blk in subspace_blocks.values()
                    )
                )
        except Exception as exc:  # pragma: no cover - robustez opcional
            base_block.setdefault("meta", {})["subspace_error"] = str(exc)
            base_block.setdefault("meta", {})["subspace_error_type"] = exc.__class__.__name__
            logger.exception("Error en subespacios para par %s", pair)

        out[pair] = base_block
        logger.log(
            level,
            "Par %s finalizado en %.6f s (bloques=%d)",
            pair,
            perf_counter() - pair_start,
            1 + len(base_block.get("subspace_variants", {})),
        )
    logger.log(level, "compute_frontier_planes_all_modes completado | pares=%d", len(out))
    return out


# -----------------------------------------------------------------------------------------------
# Visualización (se reutiliza la versión previa)
# -----------------------------------------------------------------------------------------------
def plot_planes_with_point_lines(
    res: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    records: Iterable[DeltaRecord],
    pair: Optional[Tuple[int, int]] = None,
    dims: Tuple[int, int, int] = (0, 1, 2),
    feature_names: Optional[Sequence[str]] = None,
    show_planes: bool = True,
    show_points: bool = True,
    line_kind: str = "both",
    plane_opacity: float = 0.28,
    point_opacity: float = 0.7,
    line_opacity: float = 0.35,
    max_points: Optional[int] = None,
    prefer_cp: bool = True,
    renderer: Optional[str] = None,
    title: str = "Planos + líneas por punto",
    show: bool = True,
    return_fig: bool = False,
):
    """
    Versión robusta:
      - Si res[pair]['assignment']['rec_indices'] está vacío o ausente,
        reconstruye A,B,F y asignación a planos desde 'records'.
    """

    import plotly.graph_objects as go
    import plotly.express as px

    keys = list(res.keys())
    if pair is None:
        if not keys:
            raise ValueError("res está vacío")
        pair = keys[0]
    if pair not in res:
        raise KeyError(f"El par {pair} no está en res; disponibles: {keys}")

    block = res[pair]
    planes_by_label = block.get("planes_by_label", {})
    if not planes_by_label:
        raise ValueError("No hay planos que graficar para este par (planes_by_label vacío).")

    def _axis_label(idx):
        if feature_names and 0 <= idx < len(feature_names):
            return feature_names[idx]
        return f"x{idx}"

    def _rgba(col, alpha):
        if isinstance(col, str) and col.startswith("#") and len(col) == 7:
            r = int(col[1:3], 16)
            g = int(col[3:5], 16)
            b = int(col[5:7], 16)
            return f"rgba({r},{g},{b},{float(alpha):.3f})"
        if isinstance(col, str) and col.startswith("rgb("):
            return col.replace("rgb(", "rgba(").replace(")", f",{float(alpha):.3f})")
        return col

    def _restrict_plane(n, b, mu_full, dims_sel):
        n = np.asarray(n, float).reshape(-1)
        mu_full = np.asarray(mu_full, float).reshape(-1)
        d = n.size
        dims_sel = tuple(int(i) for i in dims_sel)
        other = [j for j in range(d) if j not in dims_sel]
        b_eff = float(b + (n[other] @ mu_full[other])) if other else float(b)
        n_sub = n[list(dims_sel)]
        return n_sub, b_eff

    def _plane_mesh(n_sub, b_eff, lo, hi, res=16):
        n_sub = np.asarray(n_sub, float).reshape(3)
        k = int(np.argmax(np.abs(n_sub)))
        free = [i for i in (0, 1, 2) if i != k]
        ax0, ax1 = free[0], free[1]
        g0 = np.linspace(lo[ax0], hi[ax0], res)
        g1 = np.linspace(lo[ax1], hi[ax1], res)
        G0, G1 = np.meshgrid(g0, g1)
        denom = n_sub[k] if abs(n_sub[k]) > 1e-12 else 1e-12
        Gk = -(n_sub[ax0] * G0 + n_sub[ax1] * G1 + b_eff) / denom
        grids = {ax0: G0, ax1: G1, k: Gk}
        return grids[0], grids[1], grids[2]

    def _canon_u(u):
        u = u.astype(float)
        n = np.linalg.norm(u) + 1e-12
        u = u / n
        if u[0] < 0:
            u = -u
        return u

    flat_n, flat_b, flat_lbl, flat_j = [], [], [], []
    mu_list = []
    for lbl, plist in planes_by_label.items():
        for j, meta in enumerate(plist):
            flat_n.append(np.asarray(meta["n"], float))
            flat_b.append(float(meta["b"]))
            flat_lbl.append(int(lbl))
            flat_j.append(int(j))
            mu_list.append(np.asarray(meta["mu"], float))
    flat_n = np.vstack(flat_n)
    flat_b = np.asarray(flat_b, float)
    flat_lbl = np.asarray(flat_lbl, int)
    flat_j = np.asarray(flat_j, int)

    assignment = block.get("assignment", {}) or {}
    rec_idx = np.asarray(assignment.get("rec_indices", []), int)
    assigned_label = np.asarray(assignment.get("assigned_label", []), int)
    assigned_plane = np.asarray(assignment.get("assigned_plane", []), int)

    A_list, B_list, F_list, idx_list = [], [], [], []
    eps = 1e-3
    a_cls, b_cls = int(pair[0]), int(pair[1])
    if rec_idx.size == 0:
        for i, r in enumerate(records):
            if not getattr(r, "success", True):
                continue
            y0, y1 = int(getattr(r, "y0")), int(getattr(r, "y1"))
            if {y0, y1} != {a_cls, b_cls}:
                continue
            a, b = _endpoints(r, prefer_cp=prefer_cp)
            v = b - a
            if np.linalg.norm(v) <= 1e-12:
                continue
            u = _canon_u(v)
            F = b + eps * u
            A_list.append(a)
            B_list.append(b)
            F_list.append(F)
            idx_list.append(i)

        if not F_list:
            raise ValueError("No pude reconstruir segmentos para este par desde 'records'.")

        A = np.vstack(A_list)
        B = np.vstack(B_list)
        F = np.vstack(F_list)
        rec_idx = np.asarray(idx_list, int)

        H = np.abs(F @ flat_n.T + flat_b[None, :])
        best = np.argmin(H, axis=1)
        assigned_label = flat_lbl[best]
        assigned_plane = flat_j[best]
    else:
        for i in rec_idx:
            r = records[i]
            a, b = _endpoints(r, prefer_cp=prefer_cp)
            v = b - a
            if np.linalg.norm(v) <= 1e-12:
                continue
            u = _canon_u(v)
            F = b + eps * u
            A_list.append(a)
            B_list.append(b)
            F_list.append(F)
        if not F_list:
            raise ValueError("No pude reconstruir segmentos para rec_indices; ¿coinciden con 'records'?")
        A = np.vstack(A_list)
        B = np.vstack(B_list)
        F = np.vstack(F_list)

    base_palette = px.colors.qualitative.Plotly

    def _base_color_for(label, j):
        base = base_palette[int(label) % len(base_palette)]
        if base.startswith("#"):
            r = int(base[1:3], 16)
            g = int(base[3:5], 16)
            b = int(base[5:7], 16)
            fac = 0.85 + 0.1 * ((int(j) % 3))
            r = int(np.clip(r * fac, 0, 255))
            g = int(np.clip(g * fac, 0, 255))
            b = int(np.clip(b * fac, 0, 255))
            base = f"rgb({r},{g},{b})"
        return base

    dims = tuple(int(i) for i in dims)
    if len(dims) != 3:
        raise ValueError("Esta función es 3D: 'dims' debe tener tres índices.")

    MU = np.vstack(mu_list) if mu_list else F
    lo = MU[:, dims].min(axis=0)
    hi = MU[:, dims].max(axis=0)
    allP = np.vstack([A[:, dims], B[:, dims], F[:, dims]])
    lo = np.minimum(lo, allP.min(axis=0))
    hi = np.maximum(hi, allP.max(axis=0))
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - 0.10 * span
    hi = hi + 0.10 * span

    n_tot = F.shape[0]
    if (max_points is not None) and (max_points > 0) and (n_tot > max_points):
        stride = int(np.ceil(n_tot / max_points))
        sel = np.arange(0, n_tot, stride, dtype=int)
    else:
        sel = np.arange(n_tot, dtype=int)

    if renderer:
        import plotly.io as pio

        try:
            pio.renderers.default = renderer
        except Exception:
            pass
    fig = go.Figure()

    if show_points:
        col_pts = _rgba("rgb(90,90,90)", point_opacity)
        P = F[sel][:, dims]
        fig.add_trace(
            go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode="markers",
                name="Frontera (F)",
                marker=dict(size=3, color=col_pts),
                hovertemplate="F<br>x:%{x:.3f}<br>y:%{y:.3f}<br>z:%{z:.3f}",
            )
        )

    if line_kind in ("segment", "both"):
        for lbl in np.unique(assigned_label):
            idx_lbl = np.flatnonzero(assigned_label == lbl)
            if idx_lbl.size == 0:
                continue
            planes_lbl = planes_by_label.get(int(lbl), [])
            for j in range(len(planes_lbl)):
                mask = (assigned_label == int(lbl)) & (assigned_plane == j)
                idx = sel[mask[sel]] if sel.size else np.array([], int)
                if idx.size == 0:
                    continue
                Ap = A[idx][:, dims]
                Fp = F[idx][:, dims]
                xs, ys, zs = [], [], []
                for i in range(idx.size):
                    xs += [Ap[i, 0], Fp[i, 0], None]
                    ys += [Ap[i, 1], Fp[i, 1], None]
                    zs += [Ap[i, 2], Fp[i, 2], None]
                col = _rgba(_base_color_for(lbl, j), line_opacity)
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=f"Segmento (label {lbl}, plano {j})",
                        legendgroup=f"lbl{lbl}-pl{j}",
                        line=dict(width=2, color=col),
                        hoverinfo="skip",
                    )
                )

    if line_kind in ("normal", "both"):
        lut = {}
        for lbl, plist in planes_by_label.items():
            for j, meta in enumerate(plist):
                lut[(int(lbl), int(j))] = (np.asarray(meta["n"], float), float(meta["b"]))
        for lbl in np.unique(assigned_label):
            planes_lbl = planes_by_label.get(int(lbl), [])
            for j in range(len(planes_lbl)):
                mask = (assigned_label == int(lbl)) & (assigned_plane == j)
                idx = sel[mask[sel]] if sel.size else np.array([], int)
                if idx.size == 0:
                    continue
                n_full, b_full = lut[(int(lbl), int(j))]
                F_full = F[idx]
                dist = F_full @ n_full + b_full
                F_proj = F_full - dist[:, None] * n_full[None, :]
                P1 = F_full[:, dims]
                P2 = F_proj[:, dims]
                xs, ys, zs = [], [], []
                for i in range(idx.size):
                    xs += [P1[i, 0], P2[i, 0], None]
                    ys += [P1[i, 1], P2[i, 1], None]
                    zs += [P1[i, 2], P2[i, 2], None]
                col = _rgba(_base_color_for(lbl, j), line_opacity)
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=f"Normal (label {lbl}, plano {j})",
                        legendgroup=f"lbl{lbl}-pl{j}",
                        line=dict(width=2, color=col),
                        hoverinfo="skip",
                    )
                )

    if show_planes:
        for lbl, plist in planes_by_label.items():
            for j, meta in enumerate(plist):
                n_full = np.asarray(meta["n"], float)
                b_full = float(meta["b"])
                mu_full = np.asarray(meta["mu"], float)
                n_sub, b_eff = _restrict_plane(n_full, b_full, mu_full, dims)
                Xs, Ys, Zs = _plane_mesh(n_sub, b_eff, lo, hi, res=16)
                col = _rgba(_base_color_for(lbl, j), 1.0)
                fig.add_trace(
                    go.Surface(
                        x=Xs,
                        y=Ys,
                        z=Zs,
                        name=f"Plano (label {lbl}, plano {j})",
                        legendgroup=f"lbl{lbl}-pl{j}",
                        showscale=False,
                        opacity=float(plane_opacity),
                        colorscale=[[0, col], [1, col]],
                        hovertemplate=(
                            f"<b>Plano label {lbl}, #{j}</b><br>"
                            f"n·x + b = 0<br>"
                            f"b={b_full:.4f}<br>"
                            f"RMSE(inliers)={meta['fit_error']['inlier_rmse']:.4f}"
                            "<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        title=title + f" — par {pair}",
        scene=dict(
            xaxis_title=_axis_label(dims[0]),
            yaxis_title=_axis_label(dims[1]),
            zaxis_title=_axis_label(dims[2]),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
    )
    if renderer:
        import plotly.io as pio

        try:
            pio.renderers.default = renderer
        except Exception:
            pass

    if show:
        fig.show(config={"scrollZoom": True, "responsive": True})
        if not return_fig:
            return None

    return fig
