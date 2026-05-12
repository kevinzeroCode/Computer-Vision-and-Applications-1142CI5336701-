import numpy as np
import cv2
import os

# ─── Camera Parameters ────────────────────────────────────────────────────────

K_TOP = np.array([
    [1440., 0.,    540.],
    [0.,    1440., 960.],
    [0.,    0.,    1.  ]
])
RT_TOP = np.array([
    [-0.9063, -0.4226,  0.0000,   0.7553],
    [-0.0552,  0.1184, -0.9914,  31.1531],
    [ 0.4190, -0.8985, -0.1307, 105.0863]
])

K_BOT = np.array([
    [1140., 0.,    540.],
    [0.,    1140., 960.],
    [0.,    0.,    1.  ]
])
RT_BOT = np.array([
    [-0.9063, -0.4226,  0.0000,  2.7223],
    [ 0.0000,  0.0000, -1.0000, 32.4138],
    [ 0.4226, -0.9063,  0.0000, 91.9004]
])

P_TOP = K_TOP @ RT_TOP  # (3,4)
P_BOT = K_BOT @ RT_BOT  # (3,4)


# ─── Fundamental Matrix ───────────────────────────────────────────────────────

def skew(v):
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])

def compute_fundamental():
    R1, t1 = RT_TOP[:, :3], RT_TOP[:, 3]
    R2, t2 = RT_BOT[:, :3], RT_BOT[:, 3]
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    E = skew(t_rel) @ R_rel
    F = np.linalg.inv(K_BOT).T @ E @ np.linalg.inv(K_TOP)
    return F


# ─── Image I/O ────────────────────────────────────────────────────────────────

def read_image(path):
    # np.fromfile handles Unicode paths that break cv2.imread on Windows
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ─── Background Model ─────────────────────────────────────────────────────────

def build_background(img_dir, step=4):
    # Median of every step-th frame; laser covers <3% of sampled frames per pixel
    frames = []
    for i in range(0, 120, step):
        img = read_image(os.path.join(img_dir, f'{i:04d}.jpg'))
        frames.append(img[:, :, 2].astype(np.float32))
    return np.median(frames, axis=0)  # (H, W) Red channel


# ─── Laser Line Extraction ────────────────────────────────────────────────────

def extract_laser(img, bg_r, diff_thresh=25.0, half_win=5):
    """Return (N,2) array of [col, subpixel_row] laser points."""
    diff = img[:, :, 2].astype(np.float32) - bg_r  # (H, W)
    H, W = diff.shape
    pts = []
    for col in range(W):
        d = diff[:, col]
        if d.max() < diff_thresh:
            continue
        peak = int(np.argmax(d))
        lo = max(0, peak - half_win)
        hi = min(H, peak + half_win + 1)
        w = np.clip(d[lo:hi], 0, None)
        total = w.sum()
        if total < 1e-6:
            continue
        rows = np.arange(lo, hi, dtype=np.float64)
        subpix_row = float((w * rows).sum() / total)
        pts.append([float(col), subpix_row])
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))


# ─── Epipolar Correspondences ─────────────────────────────────────────────────

def find_correspondences(top_pts, bot_pts, F, epi_thresh=3.0):
    """For each TOP point find the nearest BOT point to its epipolar line."""
    N = len(top_pts)
    top_h = np.c_[top_pts, np.ones(N)]               # (N,3)
    bot_h = np.c_[bot_pts, np.ones(len(bot_pts))]    # (M,3)
    lines = (F @ top_h.T).T                           # (N,3)
    norms = np.maximum(np.sqrt(lines[:, 0]**2 + lines[:, 1]**2), 1e-10)
    dists = np.abs(bot_h @ lines.T) / norms           # (M,N)
    best_idx = np.argmin(dists, axis=0)               # (N,)
    best_d = dists[best_idx, np.arange(N)]
    mask = best_d < epi_thresh
    return top_pts[mask], bot_pts[best_idx[mask]]


# ─── Triangulation ────────────────────────────────────────────────────────────

def triangulate(m_top, m_bot):
    """DLT triangulation; returns (pts3D, valid_m_top, valid_m_bot)."""
    pts4D = cv2.triangulatePoints(
        P_TOP.astype(np.float64),
        P_BOT.astype(np.float64),
        m_top.T.astype(np.float32),
        m_bot.T.astype(np.float32)
    )
    w = pts4D[3]
    valid = np.abs(w) > 1e-6
    pts3D = (pts4D[:3, valid] / w[valid]).T
    return pts3D, m_top[valid], m_bot[valid]


# ─── Outlier Rejection ────────────────────────────────────────────────────────

def reject_outliers(pts3D, m_top, m_bot, thresh=2.0):
    """Remove points whose reprojection error exceeds thresh in either camera."""
    pts_h = np.c_[pts3D, np.ones(len(pts3D))]
    mask = np.ones(len(pts3D), dtype=bool)
    for P, obs in [(P_TOP, m_top), (P_BOT, m_bot)]:
        proj = (P @ pts_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        errs = np.linalg.norm(proj - obs, axis=1)
        mask &= errs < thresh
    return pts3D[mask]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    F = compute_fundamental()

    print('Building background models...')
    bg_top = build_background('TOP')
    bg_bot = build_background('BOT')

    all_pts = []
    for i in range(120):
        top_img = read_image(f'TOP/{i:04d}.jpg')
        bot_img = read_image(f'BOT/{i:04d}.jpg')

        top_pts = extract_laser(top_img, bg_top)
        bot_pts = extract_laser(bot_img, bg_bot)

        if len(top_pts) == 0 or len(bot_pts) == 0:
            print(f'Frame {i:04d}: no laser detected (top={len(top_pts)}, bot={len(bot_pts)})')
            continue

        m_top, m_bot = find_correspondences(top_pts, bot_pts, F)
        if len(m_top) == 0:
            print(f'Frame {i:04d}: no correspondences found')
            continue

        pts3D, m_top_v, m_bot_v = triangulate(m_top, m_bot)
        pts3D = reject_outliers(pts3D, m_top_v, m_bot_v)
        all_pts.append(pts3D)
        print(f'Frame {i:04d}: {len(pts3D):4d} points')

    if not all_pts:
        print('ERROR: No 3D points reconstructed.')
        return

    cloud = np.vstack(all_pts)

    # Remove statistical outliers: points outside Q1-3*IQR to Q3+3*IQR per axis
    mask = np.ones(len(cloud), dtype=bool)
    for axis in range(3):
        q1, q3 = np.percentile(cloud[:, axis], [25, 75])
        iqr = q3 - q1
        mask &= (cloud[:, axis] >= q1 - 3 * iqr) & (cloud[:, axis] <= q3 + 3 * iqr)
    n_before = len(cloud)
    cloud = cloud[mask]
    print(f'Outlier removal: {n_before - len(cloud)} removed, {len(cloud)} kept')

    np.savetxt('M11415015.xyz', cloud, fmt='%.6f')
    print(f'Total: {len(cloud)} points saved to M11415015.xyz')


if __name__ == '__main__':
    main()
