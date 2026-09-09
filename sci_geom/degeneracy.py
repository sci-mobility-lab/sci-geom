"""Is the along-track direction actually constrained by the geometry, or only apparently?

⚠️ **The failure this exists to catch.** A corridor is close to a straight line. Point-to-plane
registration is constrained by surface normals, and on a corridor almost every normal points
sideways (walls, embankments, treelines) or upward (the road). Almost none points *along* the
corridor. So the along-track direction is weakly constrained -- the classic degeneracy, documented
for exactly this case:

    "In corridor sections, the ICP registration is expected to degenerate in one axis along the
     corridor."   -- X-ICP, IEEE T-RO 2023 (arXiv:2211.16335); see also LP-ICP (arXiv:2501.02580)

Why this matters more here than in most robotics work: the residual error lands on **station**,
the along-corridor coordinate, and station is the join key between the LiDAR map and every gaze
measurement. An aggregate RMSE hides it completely -- lateral and vertical are well constrained, so
the total looks fine while the one axis we depend on is not.

**We do not fix the degeneracy. We detect and report it**, so `station_sigma_m` is honest instead
of assumed. A wide station uncertainty is a usable result; a narrow one that is wrong is not.

Method: the point-to-plane translational information matrix

    A = sum_i  n_i n_i^T          (3x3, from surface normals)

Its eigenvalues are the constraint strength along each principal direction. The along-track
constraint is `d^T A d` for the unit corridor direction `d`. Normalising by the strongest direction
gives a dimensionless **localizability ratio** in [0, 1]: 1 means as well constrained as the best
axis, ~0 means unobservable. This is the quantity X-ICP calls localizability; here it is used only
as a diagnostic and to size an uncertainty, never to reweight a solver.
"""
from __future__ import annotations

import numpy as np


def normals_from_neighbourhood(points, k=16):
    """Per-point surface normals by local PCA. Small k, because we want local surfaces."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if len(points) < k:
        raise ValueError(f"need at least k={k} points, got {len(points)}")
    _, idx = cKDTree(points).query(points, k=k)
    nbr = points[idx]                              # (N, k, 3)
    nbr = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbr, nbr) / k
    # smallest-eigenvector of the local covariance is the surface normal
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def information_matrix(normals):
    """Point-to-plane translational information matrix, A = sum n n^T."""
    n = np.asarray(normals, dtype=np.float64)
    return n.T @ n


def localizability(normals, direction):
    """Dimensionless constraint strength along `direction`, in [0, 1].

    1.0 = as well constrained as the best-constrained axis. Near 0 = unobservable.
    """
    A = information_matrix(normals)
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    strongest = float(np.linalg.eigvalsh(A).max())
    if strongest <= 0:
        return 0.0
    return float((d @ A @ d) / strongest)


def station_sigma_from_localizability(ratio, *, sigma_best_m=0.05, sigma_max_m=5.0):
    """Turn a localizability ratio into a station standard deviation, in metres.

    Information scales as 1/sigma^2, so sigma scales as 1/sqrt(information):

        sigma = sigma_best / sqrt(ratio)

    ⚠️ `sigma_best_m` is the uncertainty a *well-constrained* axis achieves on this rig, and it must
    be measured against surveyed checkpoints -- it is NOT derivable from the point cloud. The
    default is a placeholder, marked ILLUSTRATIVE ONLY, in the same spirit as every other
    unvalidated threshold in this project. `sigma_max_m` caps the result so a fully degenerate
    section reports a large-but-finite number rather than infinity.
    """
    ratio = float(ratio)
    if ratio <= 0:
        return float(sigma_max_m)
    return float(min(sigma_max_m, sigma_best_m / np.sqrt(ratio)))


def corridor_profile(points, corridor_xyz, *, window_m=25.0, step_m=25.0,
                     k=16, max_points_per_window=20000, rng_seed=0):
    """Walk the corridor and report localizability along it, station by station.

    Returns a list of dicts: station_m, along_ratio, lateral_ratio, vertical_ratio,
    station_sigma_m, n_points.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    corridor_xyz = np.asarray(corridor_xyz, dtype=np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(corridor_xyz[:, :2], axis=0), axis=1))]
    tree = cKDTree(points[:, :2])
    rng = np.random.default_rng(rng_seed)

    out = []
    for s in np.arange(arc[0], arc[-1], step_m):
        i = int(np.searchsorted(arc, s))
        i = min(max(i, 1), len(corridor_xyz) - 2)
        centre = corridor_xyz[i]
        along = corridor_xyz[i + 1][:2] - corridor_xyz[i - 1][:2]
        na = np.linalg.norm(along)
        if na < 1e-6:
            continue
        along = np.array([along[0] / na, along[1] / na, 0.0])
        lateral = np.array([-along[1], along[0], 0.0])
        vertical = np.array([0.0, 0.0, 1.0])

        idx = tree.query_ball_point(centre[:2], window_m)
        if len(idx) < 4 * k:
            continue
        if len(idx) > max_points_per_window:
            idx = rng.choice(idx, max_points_per_window, replace=False)
        win = points[np.asarray(idx)]
        nrm = normals_from_neighbourhood(win, k=k)

        a = localizability(nrm, along)
        out.append(dict(
            station_m=float(s),
            along_ratio=a,
            lateral_ratio=localizability(nrm, lateral),
            vertical_ratio=localizability(nrm, vertical),
            station_sigma_m=station_sigma_from_localizability(a),
            n_points=int(len(win)),
        ))
    return out
