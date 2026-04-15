"""
estimate_projection.py
Estimate 3x4 projection matrices P from 2D/3D point correspondences using DLT.

Key insight:
  - Sort 3D pts by z (height) descending → match to 2D pts sorted by v (image row) ascending
  - Within z-groups of similar height, try all x!-orderings using ±x correlation with u
  - Try both camera orientations (gnome's right = left vs right in image)
  - For count mismatches (N_3d = N_2d+1), try dropping each 3D pt

Usage:
    python estimate_projection.py

Outputs:
    matrices/P01.npy ... P07.npy
"""

import os
import numpy as np
from itertools import permutations
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

N_IMAGES    = 7
POINTS_DIR  = 'points'
MATRICES_DIR = 'matrices'
Z_GROUP_TOL = 4.0   # 3D pts within this many units in z are treated as a group
os.makedirs(MATRICES_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  DLT core                                                                    #
# --------------------------------------------------------------------------- #

def dlt(pts3d, pts2d):
    N = len(pts3d)
    assert N >= 6
    mean3  = pts3d.mean(axis=0)
    scale3 = np.sqrt(3) / (np.linalg.norm(pts3d - mean3, axis=1).mean() + 1e-12)
    T3 = np.diag([scale3, scale3, scale3, 1.0])
    T3[:3, 3] = -scale3 * mean3

    mean2  = pts2d.mean(axis=0)
    scale2 = np.sqrt(2) / (np.linalg.norm(pts2d - mean2, axis=1).mean() + 1e-12)
    T2 = np.array([[scale2, 0, -scale2*mean2[0]],
                   [0, scale2, -scale2*mean2[1]],
                   [0, 0, 1.0]])

    p3n = (T3 @ np.hstack([pts3d, np.ones((N,1))]).T).T
    p2n = (T2 @ np.hstack([pts2d, np.ones((N,1))]).T).T

    A = []
    for i in range(N):
        X, Y, Z, W = p3n[i];  u, v, _ = p2n[i]
        A.append([X,Y,Z,W, 0,0,0,0, -u*X,-u*Y,-u*Z,-u*W])
        A.append([0,0,0,0, X,Y,Z,W, -v*X,-v*Y,-v*Z,-v*W])

    _, _, Vt = np.linalg.svd(np.array(A))
    P_n = Vt[-1].reshape(3, 4)
    P   = np.linalg.inv(T2) @ P_n @ T3
    # Sign fix: flip P so that the majority of input points project in front
    pts_h  = np.hstack([pts3d, np.ones((N, 1))])
    depths = (P[2] @ pts_h.T)          # (N,) camera-space z for each point
    if np.median(depths) < 0:
        P = -P
    return P


def reproj_err(P, pts3d, pts2d):
    N  = len(pts3d)
    h  = np.hstack([pts3d, np.ones((N, 1))])
    pr = (P @ h.T).T
    if np.any(pr[:, 2] <= 0):
        return np.full(N, 1e9), 1e9
    uv = pr[:, :2] / pr[:, 2:3]
    e  = np.linalg.norm(uv - pts2d, axis=1)
    return e, e.mean()


def reproj_err_robust(P, pts3d, pts2d):
    """Like reproj_err but skips negative-depth points (returns 1e6 for them)."""
    N  = len(pts3d)
    h  = np.hstack([pts3d, np.ones((N, 1))])
    pr = (P @ h.T).T
    pz = pr[:, 2]
    valid = pz > 0
    if valid.sum() < 4:
        return np.full(N, 1e9), 1e9
    e = np.full(N, 1e6)
    uv = pr[valid, :2] / pz[valid, None]
    e[valid] = np.linalg.norm(uv - pts2d[valid], axis=1)
    return e, e[valid].mean()


def dlt_ransac(pts3d, pts2d, names, tag):
    """Try DLT on full set and all drop-one subsets; return best valid P."""
    N = len(pts3d)
    best_P, best_err, best_drop = None, np.inf, -1

    # Full set
    try:
        P = dlt(pts3d, pts2d)
        if is_valid_camera(P, pts3d, tag):
            _, me = reproj_err_robust(P, pts3d, pts2d)
            if me < best_err:
                best_P, best_err, best_drop = P, me, -1
    except Exception:
        pass

    # Drop-one
    for drop in range(N):
        sub3 = np.delete(pts3d, drop, axis=0)
        sub2 = np.delete(pts2d, drop, axis=0)
        if len(sub3) < 6:
            continue
        try:
            P = dlt(sub3, sub2)
            if not is_valid_camera(P, sub3, tag):
                continue
            _, me = reproj_err_robust(P, sub3, sub2)
            if me < best_err:
                best_P, best_err, best_drop = P, me, drop
        except Exception:
            pass

    if best_drop >= 0:
        dropped_name = names[best_drop] if names else str(best_drop)
        print(f'  DLT RANSAC: best drop = [{best_drop}] {dropped_name}  err={best_err:.2f} px')
    else:
        print(f'  DLT RANSAC: no drop needed  err={best_err:.2f} px')
    return best_P, best_drop, best_err


def solvepnp_best(pts3d, pts2d, tag=None, img_w=1836, img_h=2748):
    """
    Try cv2.solvePnPRansac with several focal-length guesses.
    Returns (P 3x4, mean_reproj_err) or (None, inf) if cv2 unavailable.
    """
    if not HAVE_CV2:
        return None, np.inf

    best_P, best_err = None, np.inf
    cx, cy = img_w / 2.0, img_h / 2.0

    for f in [2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0,  1]], dtype=np.float64)
        for thresh in [8.0, 50.0, 150.0, 300.0]:
            try:
                ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d.astype(np.float64),
                    pts2d.astype(np.float64),
                    K, None,
                    iterationsCount=2000,
                    reprojectionError=thresh,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            except Exception:
                continue
            if not ret or inliers is None or len(inliers) < 6:
                continue
            R, _ = cv2.Rodrigues(rvec)
            Rt = np.hstack([R, tvec])
            P  = K @ Rt
            if not is_valid_camera(P, pts3d, tag):
                continue
            # Score = mean error over ALL points (same basis as DLT scoring)
            _, me = reproj_err_robust(P, pts3d, pts2d)
            if me < best_err:
                best_P, best_err = P, me

    return best_P, best_err


def camera_center(P):
    return -np.linalg.inv(P[:, :3]) @ P[:, 3]


# Per-image camera direction hints (sign of X, Y relative to model centroid)
# None means unconstrained.  +1 = positive direction,  -1 = negative direction
# Based on known camera positions from ground-truth inspection.
CAMERA_HINTS = {
    '01': dict(x=+1),           # right side
    '02': dict(x=-1),           # left side
    '03': dict(x=+1),           # right side (front-ish)
    '04': dict(x=-1),           # left side (can see welcome signs → not pure back)
    '05': dict(y=+1),           # front
    '06': dict(x=-1),           # left side
    '07': dict(x=+1, y=-1),    # right-back
}

def is_valid_camera(P, pts3d, tag=None):
    C    = camera_center(P)
    cent = pts3d.mean(axis=0)
    d    = np.linalg.norm(C - cent)
    if not (40 < d < 3000):
        return False
    # Check per-image direction hint if provided
    if tag and tag in CAMERA_HINTS:
        hints = CAMERA_HINTS[tag]
        rel = C - cent   # camera position relative to model centroid
        if 'x' in hints and np.sign(rel[0]) != hints['x']:
            return False
        if 'y' in hints and np.sign(rel[1]) != hints['y']:
            return False
    return True


# --------------------------------------------------------------------------- #
#  Group-aware permutation search                                              #
# --------------------------------------------------------------------------- #

def get_z_groups(pts3d, tol=Z_GROUP_TOL):
    """
    Sort pts by z descending. Group consecutive pts with z-diff < tol.
    Returns list of groups, each group is a list of original indices.
    """
    order = np.argsort(-pts3d[:, 2])
    groups = []
    cur = [order[0]]
    for idx in order[1:]:
        if pts3d[cur[-1], 2] - pts3d[idx, 2] < tol:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)
    return groups  # list of lists; each list = original 3D indices in one height group


def build_permutations(pts3d, pts2d):
    """
    Generate all candidate permutations using z→v group assignment.
    Steps:
      1. Sort 3D pts by z desc → match to 2D pts sorted by v asc (group level)
      2. Within each z-group, try ALL n! orderings of the assigned 2D pts
    Total candidates = product over groups of (group_size!)
    """
    N = len(pts3d)
    v_order = np.argsort(pts2d[:, 1])   # 2D indices: top→bottom in image
    groups  = get_z_groups(pts3d)

    # Assign v-slots to each group
    v_slots = []
    slot = 0
    for g in groups:
        v_slots.append(list(v_order[slot:slot + len(g)]))
        slot += len(g)

    def recurse(g_idx, partial_perm):
        if g_idx == len(groups):
            yield partial_perm.copy()
            return
        g3 = groups[g_idx]      # 3D indices in this height group
        vs = v_slots[g_idx]     # 2D indices assigned to this group
        n  = len(g3)
        if n == 1:
            partial_perm[g3[0]] = vs[0]
            yield from recurse(g_idx + 1, partial_perm)
        else:
            # Try ALL n! orderings of the 2D slots for the 3D pts in this group
            for vperm in permutations(vs):
                for k, g3i in enumerate(g3):
                    partial_perm[g3i] = vperm[k]
                yield from recurse(g_idx + 1, partial_perm)

    perm = np.zeros(N, dtype=int)
    yield from recurse(0, perm)


def find_best_perm(pts3d, pts2d, tag=''):
    # Extract base tag (e.g. '06' from '06 drop#2')
    base_tag = tag.split()[0] if tag else None
    N = len(pts3d)
    best_err = np.inf
    best_P   = None
    best_ord = list(range(N))

    n_tried = 0
    for perm in build_permutations(pts3d, pts2d):
        p2 = pts2d[perm]
        n_tried += 1
        try:
            P = dlt(pts3d, p2)
            if not is_valid_camera(P, pts3d, base_tag):
                continue
            _, me = reproj_err(P, pts3d, p2)
            if me < best_err:
                best_err = me
                best_P   = P
                best_ord = perm.copy()
        except Exception:
            pass

    print(f'  [{tag}] Tried {n_tried} candidate permutations  (N={N})')
    return best_P, best_ord, best_err


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def read_named_header(path):
    """Return list of landmark names from first comment line, or [] if none.
    Handles both formats:
      '# hat_tip right_eye ...'              (2D files, space-separated)
      '# 3D pts for imgXX: hat_tip,eye,...'  (3D files, colon prefix + comma-sep)
    """
    with open(path) as fh:
        line = fh.readline().strip()
    if not line.startswith('#'):
        return []
    content = line.lstrip('#').strip()
    # Strip optional "3D pts for imgXX:" prefix
    if ':' in content:
        content = content.split(':', 1)[1].strip()
    # Accept both comma and space as separators
    tokens = content.replace(',', ' ').split()
    # Filter out pure numeric or generic label tokens
    skip = {'u', 'v', 'X', 'Y', 'Z', 'pts', 'for', '3D'}
    names = [t for t in tokens
             if t not in skip and not t.replace('.','').replace('-','').isdigit()]
    return names


def main():
    print(f'Estimating projection matrices for {N_IMAGES} images\n')

    for i in range(1, N_IMAGES + 1):
        tag = f'{i:02d}'
        f3d = os.path.join(POINTS_DIR, f'img{tag}_3d.txt')
        f2d = os.path.join(POINTS_DIR, f'img{tag}_2d.txt')
        out = os.path.join(MATRICES_DIR, f'P{tag}.npy')

        if not os.path.exists(f3d) or not os.path.exists(f2d):
            print(f'[SKIP] img{tag}: missing point files')
            continue

        pts3d = np.loadtxt(f3d, comments='#')
        pts2d = np.loadtxt(f2d, comments='#')
        if pts3d.ndim == 1: pts3d = pts3d[None, :]
        if pts2d.ndim == 1: pts2d = pts2d[None, :]
        pts3d = pts3d[:, :3];  pts2d = pts2d[:, :2]

        n3, n2 = len(pts3d), len(pts2d)
        print(f'=== img{tag}: {n3} 3D pts, {n2} 2D pts ===')

        # ── Named-correspondence mode ──────────────────────────────────────── #
        # If both files have matching landmark names in their headers,
        # the order is already correct — use RANSAC drop-one for robustness.
        names3 = read_named_header(f3d)
        names2 = read_named_header(f2d)
        if names3 and names2 and names3 == names2 and n3 == n2:
            print(f'  Named mode ({n3} pts): trying DLT-RANSAC then solvePnP ...')

            # ① DLT drop-one
            P_dlt, drop_dlt, err_dlt = dlt_ransac(pts3d, pts2d, names2, tag)

            # ② solvePnPRansac — always try, pick whichever gives lower error
            P_pnp, err_pnp = None, np.inf
            if True:
                P_pnp, err_pnp = solvepnp_best(pts3d, pts2d, tag=tag)
                if P_pnp is not None:
                    print(f'  solvePnP: err={err_pnp:.2f} px')

            # Pick better
            if P_pnp is not None and err_pnp < err_dlt:
                P, best_method = P_pnp, 'solvePnP'
                pts3d_r, pts2d_r, names_r = pts3d, pts2d, names2
            elif P_dlt is not None:
                P, best_method = P_dlt, 'DLT'
                if drop_dlt >= 0:
                    pts3d_r = np.delete(pts3d, drop_dlt, axis=0)
                    pts2d_r = np.delete(pts2d, drop_dlt, axis=0)
                    names_r = [n for k, n in enumerate(names2) if k != drop_dlt]
                else:
                    pts3d_r, pts2d_r, names_r = pts3d, pts2d, names2
            else:
                print(f'  [FAIL] No valid camera found\n')
                continue

            errs, mean_err = reproj_err_robust(P, pts3d_r, pts2d_r)
            C = camera_center(P)
            np.save(out, P)
            print(f'  [{best_method}] mean reproj err = {mean_err:.2f} px')
            print(f'  camera center ≈ ({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f})')
            for j, (e, n) in enumerate(zip(errs, names_r)):
                flag = '  <-- HIGH' if e > 10 else ''
                print(f'    {n}: {e:.2f} px{flag}')
            print(f'  Saved → {out}\n')
            continue

        # ── Permutation search (unnamed files) ────────────────────────────── #
        best_P   = None
        best_err = np.inf
        best_drop = -1
        best_ord  = None

        if n3 == n2:
            P, ord_, err = find_best_perm(pts3d, pts2d, tag)
            best_P, best_err, best_ord, best_drop = P, err, ord_, -1

        elif n3 == n2 + 1:
            print(f'  Count mismatch: trying drop-one ...')
            for drop in range(n3):
                sub3 = np.delete(pts3d, drop, axis=0)
                if len(sub3) < 6:
                    continue
                P, ord_, err = find_best_perm(sub3, pts2d, f'{tag} drop#{drop}')
                print(f'    drop 3D[{drop}]: mean err = {err:.2f} px')
                if err < best_err:
                    best_err  = err
                    best_P    = P
                    best_ord  = ord_
                    best_drop = drop

        else:
            n = min(n3, n2)
            if n < 6:
                print(f'  [FAIL] Only {n} points — need ≥6'); continue
            pts3d = pts3d[:n];  pts2d = pts2d[:n]
            P, ord_, err = find_best_perm(pts3d, pts2d, tag)
            best_P, best_err, best_ord, best_drop = P, err, ord_, -1

        if best_P is None:
            print(f'  [FAIL] No valid camera found for img{tag}\n')
            continue

        if best_drop >= 0:
            pts3d_used = np.delete(pts3d, best_drop, axis=0)
            print(f'  Best: dropped 3D[{best_drop}]')
        else:
            pts3d_used = pts3d
        pts2d_used = pts2d[best_ord]

        errs, mean_err = reproj_err(best_P, pts3d_used, pts2d_used)
        C = camera_center(best_P)

        np.save(out, best_P)
        print(f'  mean reproj err = {mean_err:.2f} px')
        print(f'  camera center ≈ ({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f})')
        for j, e in enumerate(errs):
            flag = '  <-- HIGH' if e > 10 else ''
            print(f'    pt{j+1}: {e:.2f} px{flag}')
        print(f'  Saved → {out}\n')


if __name__ == '__main__':
    main()
