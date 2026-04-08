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
    if P[2, 3] < 0:
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


def camera_center(P):
    return -np.linalg.inv(P[:, :3]) @ P[:, 3]


def is_valid_camera(P, pts3d):
    C    = camera_center(P)
    cent = pts3d.mean(axis=0)
    d    = np.linalg.norm(C - cent)
    return 10 < d < 3000


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
            if not is_valid_camera(P, pts3d):
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
            # Truncate to min count
            n = min(n3, n2)
            if n < 6:
                print(f'  [FAIL] Only {n} points — need ≥6'); continue
            pts3d = pts3d[:n];  pts2d = pts2d[:n]
            P, ord_, err = find_best_perm(pts3d, pts2d, tag)
            best_P, best_err, best_ord, best_drop = P, err, ord_, -1

        if best_P is None:
            print(f'  [FAIL] No valid camera found for img{tag}\n')
            continue

        # Final error report
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
