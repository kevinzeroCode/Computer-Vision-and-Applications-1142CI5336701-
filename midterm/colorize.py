"""
colorize.py
Main colorization pipeline.

Steps:
  1. Load Santa.xyz (x y z nx ny nz)
  2. Load projection matrices P01..P07.npy
  3. For each vertex: project to all images, back-face cull, z-buffer occlusion, sample color
  4. Blend visible colors weighted by viewing angle
  5. Output Santa_colored.ply and Santa_colored.xyz

Usage:
    python colorize.py [--zbuf_eps 0.4] [--zbuf_scale 4] [--weight_gamma 3.0]
                       [--winner_ratio 1.35] [--gray 128]
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

XYZ_FILE   = '7Images and xyz/Santa.xyz'
IMG_DIR    = '7Images and xyz'
MAT_DIR    = 'matrices'
PLY_TMPL   = 'Supplementary/SantaTriangle4Test.ply'
OUT_PLY    = 'Santa_colored.ply'
OUT_XYZ    = 'Santa_colored.xyz'
N_IMAGES   = 7


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def load_xyz(path):
    data = np.loadtxt(path)
    pts  = data[:, :3].astype(np.float64)    # (N,3)
    nors = data[:, 3:6].astype(np.float64)   # (N,3)
    return pts, nors


def load_matrices(mat_dir, n):
    Ps = []
    for i in range(1, n + 1):
        p = os.path.join(mat_dir, f'P{i:02d}.npy')
        if not os.path.exists(p):
            print(f'[WARN] {p} not found — skipping image {i}')
            Ps.append(None)
        else:
            Ps.append(np.load(p))
    return Ps


def camera_center(P):
    return -np.linalg.inv(P[:, :3]) @ P[:, 3]


def bilinear_sample(img_arr, u, v):
    """Bilinear interpolation. img_arr: (H,W,3) uint8. Returns float RGB."""
    h, w = img_arr.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0, y0 = int(u), int(v)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = u - x0, v - y0
    c = (img_arr[y0, x0] * (1 - dx) * (1 - dy)
       + img_arr[y0, x1] *      dx  * (1 - dy)
       + img_arr[y1, x0] * (1 - dx) *      dy
       + img_arr[y1, x1] *      dx  *      dy)
    return c.astype(np.float64)


def build_zbuffer(P, pts, img_h, img_w, scale=1):
    """
    Rasterize Euclidean distances (camera→vertex) into a depth buffer.
    Using actual distance avoids P-scale dependency.
    Returns depth_map (float, inf where empty).
    """
    sh, sw = img_h // scale, img_w // scale
    zbuf = np.full((sh, sw), np.inf, dtype=np.float64)

    C = camera_center(P)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj  = (P @ pts_h.T).T          # (N,3)
    valid = proj[:, 2] > 0
    proj  = proj[valid]
    pts_v = pts[valid]

    u = proj[:, 0] / proj[:, 2] / scale
    v = proj[:, 1] / proj[:, 2] / scale
    d = np.linalg.norm(pts_v - C, axis=1)   # Euclidean distance

    ui = u.astype(int)
    vi = v.astype(int)
    in_bounds = (ui >= 0) & (ui < sw) & (vi >= 0) & (vi < sh)

    for k in np.where(in_bounds)[0]:
        if d[k] < zbuf[vi[k], ui[k]]:
            zbuf[vi[k], ui[k]] = d[k]

    return zbuf, scale


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zbuf_eps', type=float, default=0.4,
                        help='Z-buffer tolerance (world units). Default 0.4')
    parser.add_argument('--zbuf_scale', type=int, default=4,
                        help='Downsample factor for z-buffer (speed/memory). Default 4')
    parser.add_argument('--weight_gamma', type=float, default=3.0,
                        help='Exponent for view-confidence weighting. Higher = sharper, less blending. Default 3.0')
    parser.add_argument('--winner_ratio', type=float, default=1.35,
                        help='Use the best view alone when its score is this much larger than the second-best. Default 1.35')
    parser.add_argument('--fallback_cos', type=float, default=-0.05,
                        help='Minimum cosine for relaxed fallback coloring. Closer to 0 is safer. Default -0.05')
    parser.add_argument('--fallback_eps_scale', type=float, default=1.5,
                        help='Multiplier for z-buffer epsilon in fallback coloring. Lower = safer. Default 1.5')
    parser.add_argument('--gray', type=int, default=128,
                        help='Fallback color for uncolored vertices')
    args = parser.parse_args()

    print('Loading Santa.xyz ...')
    pts, nors = load_xyz(XYZ_FILE)
    N = len(pts)
    print(f'  {N} vertices loaded')

    # Normalize normals
    nors_len = np.linalg.norm(nors, axis=1, keepdims=True)
    nors_len[nors_len < 1e-12] = 1.0
    nors = nors / nors_len

    print('Loading projection matrices ...')
    Ps = load_matrices(MAT_DIR, N_IMAGES)
    # Normalize each P so depth values are proportional to real distance
    # (||P[2,:3]|| = 1 is the standard metric-scale normalization)
    for i, P in enumerate(Ps):
        if P is not None:
            scale_p = np.linalg.norm(P[2, :3])
            if scale_p > 0:
                Ps[i] = P / scale_p

    print('Loading images ...')
    imgs = []
    for i in range(1, N_IMAGES + 1):
        p = os.path.join(IMG_DIR, f'{i:02d}.jpg')
        if not os.path.exists(p):
            print(f'[WARN] {p} not found')
            imgs.append(None)
        else:
            imgs.append(np.array(Image.open(p)))
    print(f'Building z-buffers (scale={args.zbuf_scale}) ...')
    zbufs = []
    for i, P in enumerate(Ps):
        if P is None or imgs[i] is None:
            zbufs.append(None)
            continue
        img_h, img_w = imgs[i].shape[:2]
        print(f'  image {i+1:02d}', end='\r', flush=True)
        zbuf, _ = build_zbuffer(P, pts, img_h, img_w, scale=args.zbuf_scale)
        zbufs.append(zbuf)
    print('  z-buffers done.     ')

    # Precompute camera centers
    centers = [camera_center(P) if P is not None else None for P in Ps]

    print('Colorizing vertices ...')
    best_score   = np.full(N, -np.inf, dtype=np.float64)
    second_score = np.full(N, -np.inf, dtype=np.float64)
    best_color   = np.zeros((N, 3), dtype=np.float64)
    second_color = np.zeros((N, 3), dtype=np.float64)

    pts_h = np.hstack([pts, np.ones((N, 1))])   # (N,4)

    for i, P in enumerate(Ps):
        if P is None or imgs[i] is None or zbufs[i] is None:
            continue
        img_arr = imgs[i]
        img_h, img_w = img_arr.shape[:2]
        zbuf    = zbufs[i]
        C       = centers[i]
        scale   = args.zbuf_scale

        print(f'  Processing image {i+1:02d} ...', end='\r', flush=True)

        # Project all vertices
        proj = (P @ pts_h.T).T          # (N,3)
        pz   = proj[:, 2]               # camera-space z (sign = front/back)
        u    = proj[:, 0] / (pz + 1e-12)
        v    = proj[:, 1] / (pz + 1e-12)

        # 1. In front of camera
        mask = pz > 0
        # 2. In image bounds
        mask &= (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        # 3. Back-face cull
        view = C[np.newaxis, :] - pts   # vertex → camera
        cos_angle = np.einsum('ij,ij->i', nors, view)
        mask &= cos_angle > 0

        idx = np.where(mask)[0]

        # Z-buffer occlusion check
        if zbuf is not None and len(idx) > 0:
            ui_z = np.clip((u[idx] / scale).astype(int), 0, zbuf.shape[1] - 1)
            vi_z = np.clip((v[idx] / scale).astype(int), 0, zbuf.shape[0] - 1)
            dist_cam = np.linalg.norm(pts[idx] - C, axis=1)
            visible  = dist_cam <= zbuf[vi_z, ui_z] + args.zbuf_eps
            idx = idx[visible]
        if len(idx) == 0:
            continue

        # Vectorised bilinear colour sampling
        u_idx = u[idx]; v_idx = v[idx]
        x0 = np.clip(u_idx.astype(int),     0, img_w - 2)
        y0 = np.clip(v_idx.astype(int),     0, img_h - 2)
        x1 = x0 + 1; y1 = y0 + 1
        dx = (u_idx - x0)[:, None]; dy = (v_idx - y0)[:, None]
        rgb = (img_arr[y0, x0] * (1-dx) * (1-dy)
             + img_arr[y0, x1] *    dx  * (1-dy)
             + img_arr[y1, x0] * (1-dx) *    dy
             + img_arr[y1, x1] *    dx  *    dy).astype(np.float64)

        # Weight = cosine of incidence angle (unnormalised view → normalise)
        view_len = np.linalg.norm(view[idx], axis=1) + 1e-12
        score = np.clip(cos_angle[idx] / view_len, 0, 1) ** args.weight_gamma

        old_best = best_score[idx].copy()
        old_best_color = best_color[idx].copy()

        better = score > old_best
        if np.any(better):
            idx_b = idx[better]
            second_score[idx_b] = old_best[better]
            second_color[idx_b] = old_best_color[better]
            best_score[idx_b] = score[better]
            best_color[idx_b] = rgb[better]

        remaining = ~better
        if np.any(remaining):
            idx_r = idx[remaining]
            score_r = score[remaining]
            second_better = score_r > second_score[idx_r]
            if np.any(second_better):
                idx_s = idx_r[second_better]
                second_score[idx_s] = score_r[second_better]
                second_color[idx_s] = rgb[remaining][second_better]

    # ── Fallback pass: color remaining vertices with relaxed back-face cull ── #
    # Handles bottom/underside vertices that no camera sees head-on.
    uncolored_mask = best_score < 0
    if uncolored_mask.any():
        fallback_score = 0.01
        for i, P in enumerate(Ps):
            if P is None or imgs[i] is None or zbufs[i] is None:
                continue
            img_arr = imgs[i]
            img_h, img_w = img_arr.shape[:2]
            zbuf    = zbufs[i]
            C       = centers[i]
            scale   = args.zbuf_scale

            proj = (P @ pts_h.T).T
            pz   = proj[:, 2]
            u    = proj[:, 0] / (pz + 1e-12)
            v    = proj[:, 1] / (pz + 1e-12)

            view = C[np.newaxis, :] - pts
            cos_angle = np.einsum('ij,ij->i', nors, view)

            # Conservative fallback: only allow points that are almost front-facing.
            mask = uncolored_mask & (pz > 0)
            mask &= (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            mask &= cos_angle > args.fallback_cos

            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue

            # Keep fallback depth test close to the primary pass to avoid painting occluded backsides.
            ui_z = np.clip((u[idx] / scale).astype(int), 0, zbuf.shape[1] - 1)
            vi_z = np.clip((v[idx] / scale).astype(int), 0, zbuf.shape[0] - 1)
            dist_cam = np.linalg.norm(pts[idx] - C, axis=1)
            visible  = dist_cam <= zbuf[vi_z, ui_z] + args.zbuf_eps * args.fallback_eps_scale
            idx = idx[visible]
            if len(idx) == 0:
                continue

            x0 = np.clip(u[idx].astype(int), 0, img_w - 2)
            y0 = np.clip(v[idx].astype(int), 0, img_h - 2)
            x1 = x0 + 1; y1 = y0 + 1
            dx = (u[idx] - x0)[:, None]; dy = (v[idx] - y0)[:, None]
            rgb = (img_arr[y0, x0] * (1-dx) * (1-dy)
                 + img_arr[y0, x1] *    dx  * (1-dy)
                 + img_arr[y1, x0] * (1-dx) *    dy
                 + img_arr[y1, x1] *    dx  *    dy).astype(np.float64)

            old_best = best_score[idx].copy()
            old_best_color = best_color[idx].copy()
            score = np.full(len(idx), fallback_score, dtype=np.float64)

            better = score > old_best
            if np.any(better):
                idx_b = idx[better]
                second_score[idx_b] = old_best[better]
                second_color[idx_b] = old_best_color[better]
                best_score[idx_b] = score[better]
                best_color[idx_b] = rgb[better]

            remaining = ~better
            if np.any(remaining):
                idx_r = idx[remaining]
                score_r = score[remaining]
                second_better = score_r > second_score[idx_r]
                if np.any(second_better):
                    idx_s = idx_r[second_better]
                    second_score[idx_s] = score_r[second_better]
                    second_color[idx_s] = rgb[remaining][second_better]

    # Normalise
    has_color = best_score > -np.inf
    colors = np.full((N, 3), args.gray, dtype=np.float64)
    if np.any(has_color):
        second_valid = second_score > -np.inf
        best_only = has_color & (~second_valid | (best_score >= args.winner_ratio * np.maximum(second_score, 1e-12)))
        colors[best_only] = best_color[best_only]

        blend_mask = has_color & ~best_only
        if np.any(blend_mask):
            pair_scores = np.stack([best_score[blend_mask], second_score[blend_mask]], axis=1)
            pair_weights = pair_scores / (pair_scores.sum(axis=1, keepdims=True) + 1e-12)
            colors[blend_mask] = (
                best_color[blend_mask] * pair_weights[:, [0]]
                + second_color[blend_mask] * pair_weights[:, [1]]
            )

    colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)
    alpha = np.where(has_color, 255, 128).astype(np.uint8)

    n_colored   = has_color.sum()
    n_uncolored = (~has_color).sum()
    print(f'\n  Colored:   {n_colored} vertices ({100*n_colored/N:.1f}%)')
    print(f'  Uncolored: {n_uncolored} vertices')

    # ------------------------------------------------------------------ #
    #  Write output XYZ                                                    #
    # ------------------------------------------------------------------ #
    print(f'Writing {OUT_XYZ} ...')
    with open(OUT_XYZ, 'w') as f:
        for k in range(N):
            x, y, z = pts[k]
            r, g, b = colors_uint8[k]
            a = alpha[k]
            f.write(f'{x} {y} {z} {r} {g} {b} {a}\n')

    # ------------------------------------------------------------------ #
    #  Write output PLY (fill template)                                   #
    # ------------------------------------------------------------------ #
    print(f'Writing {OUT_PLY} ...')
    with open(PLY_TMPL, 'r') as f:
        tmpl_lines = f.readlines()

    insert_idx = None
    for li, line in enumerate(tmpl_lines):
        if '[INSERT YOUR RESULT HERE' in line:
            insert_idx = li
            break

    if insert_idx is None:
        print('[WARN] Could not find INSERT marker in PLY template. '
              'Appending vertices at the top of data section.')
        # fallback: write all vertices after header
        header_end = next(i for i, l in enumerate(tmpl_lines) if 'end_header' in l) + 1
        insert_idx = header_end

    vertex_lines = []
    for k in range(N):
        x, y, z = pts[k]
        r, g, b = colors_uint8[k]
        a = alpha[k]
        vertex_lines.append(f'{x} {y} {z} {r} {g} {b} {a}\n')

    out_lines = (tmpl_lines[:insert_idx]
                 + vertex_lines
                 + tmpl_lines[insert_idx + 1:])

    with open(OUT_PLY, 'w') as f:
        f.writelines(out_lines)

    print(f'\nDone!\n  → {OUT_XYZ}\n  → {OUT_PLY}')
    print('\nOpen Santa_colored.ply in MeshLab to inspect the result.')


if __name__ == '__main__':
    main()
