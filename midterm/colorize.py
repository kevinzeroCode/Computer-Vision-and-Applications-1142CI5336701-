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
    python colorize.py [--zbuf_sigma 2.0] [--gray 128]
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
    parser.add_argument('--zbuf_eps', type=float, default=0.05,
                        help='Z-buffer tolerance (world units). Default 0.05')
    parser.add_argument('--zbuf_scale', type=int, default=4,
                        help='Downsample factor for z-buffer (speed/memory). Default 4')
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
    img_h, img_w = imgs[0].shape[:2] if imgs[0] is not None else (2748, 1836)

    print(f'Building z-buffers (scale={args.zbuf_scale}) ...')
    zbufs = []
    for i, P in enumerate(Ps):
        if P is None or imgs[i] is None:
            zbufs.append(None)
            continue
        print(f'  image {i+1:02d}', end='\r', flush=True)
        zbuf, _ = build_zbuffer(P, pts, img_h, img_w, scale=args.zbuf_scale)
        zbufs.append(zbuf)
    print('  z-buffers done.     ')

    # Precompute camera centers
    centers = [camera_center(P) if P is not None else None for P in Ps]

    print('Colorizing vertices ...')
    color_acc  = np.zeros((N, 3), dtype=np.float64)   # weighted colour sum
    weight_acc = np.zeros(N,      dtype=np.float64)   # total weight per vertex

    pts_h = np.hstack([pts, np.ones((N, 1))])   # (N,4)

    for i, P in enumerate(Ps):
        if P is None or imgs[i] is None or zbufs[i] is None:
            continue
        img_arr = imgs[i]
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

        # Z-buffer: skip for sparse point cloud (backface culling handles occlusion)

        idx = np.where(mask)[0]
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
        w = np.clip(cos_angle[idx] / view_len, 0, 1)[:, None]

        np.add.at(color_acc,  idx, rgb * w)
        np.add.at(weight_acc, idx, w[:, 0])

    # Normalise
    has_color = weight_acc > 0
    colors = np.full((N, 3), args.gray, dtype=np.float64)
    colors[has_color] = color_acc[has_color] / weight_acc[has_color, np.newaxis]

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
