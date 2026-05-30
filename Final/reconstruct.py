"""
reconstruct.py
3-D reconstruction from the slit-laser turntable scan, by laser-plane
triangulation.

For every frame i (turntable rotated theta_i = +2 deg * i, CCW about +z):
  1. Isolate the green laser stripe by background subtraction, using the
     excess-green response  resp = 2G - R - B  of (withLaser - withoutLaser)
     (more sensitive than G-max(R,B) on the red hat).
  2. Per image row, take the sub-pixel column of the stripe (intensity-weighted
     centroid around the peak).
  3. Back-project each stripe pixel to a camera ray and intersect it with the
     fixed laser plane  a*x + b*y = 0  ->  3-D point in the world frame.
  4. Sample its colour from withoutLaser (no green tint).
  5. Un-rotate by Rz(-theta_i) into the frame-0 object frame and accumulate.

Cleanup: volume crop -> voxel down-sample -> kNN-distance outlier removal
-> radius-outlier removal (removes the faint laser spray above the hat tip).

Output: M11415015.xyz   (x y z r g b)   and   M11415015.ply
Calibration (camera pose, focal, laser plane) comes from calib.npz.
NOTE: the ground truth is NEVER read here.
"""

import numpy as np
import cv2
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WITHLASER = os.path.join(HERE, 'ScandatawithLaser')
WITHOUT   = os.path.join(WITHLASER, 'withoutLaser')
STUDENT_ID = 'M11415015'

N_FRAMES   = 180
DEG_PER_FRAME = 2.0
ROT_SIGN   = +1.0          # +2 deg/frame CCW about +z (measured from images)
LASER_THRESH = 35.0        # excess-green response threshold
HALF_WIN   = 5             # sub-pixel centroid half-window (px)
R_MAX      = 26.0          # scan-volume radius (mm)
Z_PAD      = 1.5           # allow points slightly outside [Z_BOT, gnome top]
Z_MAX      = 72.0
VOXEL      = 0.4           # mm, down-sample leaf size
KNN_K      = 8
KNN_SIGMA  = 3.0           # drop pts with mean-kNN-dist > median + KNN_SIGMA*MAD


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ─── Laser stripe extraction (sub-pixel, per row) ───────────────────────────────
def extract_stripe(with_l, without_l):
    """Return (M,2) array of [u_subpixel, v] laser-stripe points.

    Excess-green response  2G - R - B  on the (laser - background) image: more
    sensitive than G-max(R,B) on the RED hat, where the green laser is largely
    absorbed (recovers ~94% of hat rows vs ~50%) while staying noise-free."""
    diff = with_l.astype(np.float32) - without_l.astype(np.float32)
    resp = 2 * diff[:, :, 1] - diff[:, :, 2] - diff[:, :, 0]         # BGR
    H, W = resp.shape
    peak_col = np.argmax(resp, axis=1)
    peak_val = resp[np.arange(H), peak_col]
    rows = np.where(peak_val > LASER_THRESH)[0]
    pts = []
    for v in rows:
        u = peak_col[v]
        lo, hi = max(0, u - HALF_WIN), min(W, u + HALF_WIN + 1)
        w = np.clip(resp[v, lo:hi], 0, None)
        s = w.sum()
        if s < 1e-6:
            continue
        us = float((w * np.arange(lo, hi)).sum() / s)
        pts.append((us, float(v)))
    return np.array(pts) if pts else np.empty((0, 2))


# ─── Triangulation: ray ∩ laser plane ──────────────────────────────────────────
def triangulate(uv, K, R, C, plane):
    """uv:(M,2) pixels -> (M,3) world points on plane a*x+b*y=0."""
    a, b = plane
    Kinv = np.linalg.inv(K)
    hom = np.c_[uv, np.ones(len(uv))]                 # (M,3)
    rays = (R.T @ (Kinv @ hom.T)).T                   # world directions (M,3)
    denom = a * rays[:, 0] + b * rays[:, 1]
    num = -(a * C[0] + b * C[1])
    good = np.abs(denom) > 1e-9
    s = np.full(len(uv), np.nan)
    s[good] = num / denom[good]
    X = C[None, :] + s[:, None] * rays
    return X, good & (s > 0)


def bilinear(img, uv):
    H, W = img.shape[:2]
    u = np.clip(uv[:, 0], 0, W - 1.001); v = np.clip(uv[:, 1], 0, H - 1.001)
    x0 = u.astype(int); y0 = v.astype(int); x1 = x0 + 1; y1 = y0 + 1
    dx = (u - x0)[:, None]; dy = (v - y0)[:, None]
    c = (img[y0, x0] * (1 - dx) * (1 - dy) + img[y0, x1] * dx * (1 - dy)
         + img[y1, x0] * (1 - dx) * dy + img[y1, x1] * dx * dy)
    return c[:, ::-1]  # BGR -> RGB


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


# ─── Cleanup ────────────────────────────────────────────────────────────────────
def voxel_downsample(pts, cols, leaf):
    keys = np.floor(pts / leaf).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx], cols[idx]


def knn_filter(pts, cols, k=KNN_K, sigma=KNN_SIGMA):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    md = d[:, 1:].mean(axis=1)
    med = np.median(md); mad = np.median(np.abs(md - med)) + 1e-9
    keep = md < med + sigma * 1.4826 * mad
    return pts[keep], cols[keep]


def radius_outlier(pts, cols, radius=1.0, min_pts=6):
    """Drop sparse points (fewer than min_pts neighbours within radius) -- removes
    the faint spray above the red hat tip that survives the kNN test."""
    from scipy.spatial import cKDTree
    cnt = cKDTree(pts).query_ball_point(pts, radius, return_length=True)
    keep = cnt >= min_pts
    return pts[keep], cols[keep]


def main():
    cal = np.load(os.path.join(HERE, 'calib.npz'))
    K, R, C, plane = cal['K'], cal['R'], cal['C'], cal['laser_plane']
    Z_BOT = float(cal['Z_BOT'])
    print(f'Laser plane {plane[0]:+.4f} x {plane[1]:+.4f} y = 0   f={float(cal["f"]):.1f}')

    all_pts, all_col = [], []
    for i in range(N_FRAMES):
        wl = read_image(os.path.join(WITHLASER, f'{i:04d}.jpg'))
        wo = read_image(os.path.join(WITHOUT,  f'{i:04d}.jpg'))
        uv = extract_stripe(wl, wo)
        if len(uv) == 0:
            continue
        X, ok = triangulate(uv, K, R, C, plane)
        X, uv = X[ok], uv[ok]
        # scan-volume crop (world frame, before un-rotation)
        r = np.hypot(X[:, 0], X[:, 1])
        m = (r <= R_MAX) & (X[:, 2] >= Z_BOT - Z_PAD) & (X[:, 2] <= Z_MAX)
        X, uv = X[m], uv[m]
        if len(X) == 0:
            continue
        rgb = bilinear(wo, uv)
        X = (rot_z(-ROT_SIGN * DEG_PER_FRAME * i) @ X.T).T   # -> object frame
        all_pts.append(X); all_col.append(rgb)
        if i % 30 == 0:
            print(f'  frame {i:3d}: {len(X):4d} pts')

    pts = np.vstack(all_pts); cols = np.vstack(all_col)
    print(f'Raw: {len(pts)} points')

    pts, cols = voxel_downsample(pts, cols, VOXEL)
    print(f'After voxel({VOXEL}mm): {len(pts)}')
    pts, cols = knn_filter(pts, cols)
    pts, cols = radius_outlier(pts, cols)
    print(f'After outlier removal: {len(pts)}')

    cols = np.clip(cols, 0, 255).astype(np.uint8)
    xyz = os.path.join(HERE, f'{STUDENT_ID}.xyz')
    np.savetxt(xyz, np.c_[pts, cols], fmt='%.4f %.4f %.4f %d %d %d')
    write_ply(os.path.join(HERE, f'{STUDENT_ID}.ply'), pts, cols)
    print(f'Saved {xyz}  and  {STUDENT_ID}.ply  ({len(pts)} points)')


def write_ply(path, pts, cols):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n')


if __name__ == '__main__':
    main()
