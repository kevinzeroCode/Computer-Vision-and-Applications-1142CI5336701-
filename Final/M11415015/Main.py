"""
pipeline.py  --  single-file pipeline for the slit-laser turntable scanner.

Merges the three stages into one runnable program:

    Stage 1  CALIBRATION   -> camera pose + focal (PnP on the turntable hexagon)
                              and the laser-plane equation  a*x + b*y = 0
    Stage 2  RECONSTRUCTION-> stripe detection + ray/plane triangulation, output
                              the coloured cloud  M11415015.xyz / .ply
    Stage 3  EVALUATION     -> compare to GroundTruth.ply (this is the ONLY stage
                              that reads the ground truth; skip with --no-eval)

Run:
    python pipeline.py                # calibrate + reconstruct + evaluate
    python pipeline.py --no-eval      # calibrate + reconstruct only

Method summary
--------------
World frame: origin = turntable centre, +Z = rotation axis. The turntable is a
regular hexagonal prism (circum-radius 25 mm, top z=+1, bottom z=-14). Each laser
pixel back-projects to a camera ray that is intersected with the fixed laser
plane to give a 3-D point; points are de-rotated by the frame angle and pooled.

Note on the focal length: the nominally provided f=720 px is inconsistent with
the imaged 50 mm turntable (best reprojection > 22 px); refining f against the
known hexagon gives f ~= 1281 px (= 720 * 16/9), reprojection RMS ~0.04 px, which
is adopted.  cx=360, cy=640 (image centre) check out exactly.
"""

import os
import argparse
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WITHLASER = os.path.join(HERE, 'ScandatawithLaser')
WITHOUT   = os.path.join(WITHLASER, 'withoutLaser')
GT_PLY    = os.path.join(HERE, 'GroundTruth.ply')
STUDENT_ID = 'M11415015'

# ── turntable model (mm) / intrinsics ──────────────────────────────────────────
R_HEX = 25.0          # circum-radius (vertex-vertex diameter 50 mm)
Z_TOP, Z_BOT = 1.0, -14.0     # top / bottom face (thickness 15 mm)
CX, CY = 360.0, 640.0         # principal point (= image centre)

# ── scan / detection / cleanup parameters ──────────────────────────────────────
N_FRAMES, DEG_PER_FRAME, ROT_SIGN = 180, 2.0, +1.0   # +2 deg/frame CCW about +z
LASER_THRESH = 35.0           # excess-green stripe threshold
PLANE_THRESH = 40.0           # threshold for the top-face stripe (plane fit)
HALF_WIN = 5                  # sub-pixel centroid half-window (px)
R_MAX, Z_PAD, Z_MAX = 26.0, 1.5, 72.0
VOXEL = 0.4
KNN_K, KNN_SIGMA = 8, 3.0
ROR_RADIUS, ROR_MIN = 1.0, 6


# ════════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════════════════════════════
def read_image(path):
    """Unicode-safe image read (cv2.imread breaks on the non-ASCII path)."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def laser_response(with_l, without_l):
    """Excess-green response  2G - R - B  of (laser - background).  More
    sensitive than G-max(R,B) on the RED hat, where the green laser is absorbed;
    0 % background false-positive rate (the static scene cancels in the diff)."""
    diff = with_l.astype(np.float32) - without_l.astype(np.float32)
    return 2 * diff[:, :, 1] - diff[:, :, 2] - diff[:, :, 0]     # BGR


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def write_ply(path, pts, cols):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n')


def _hex_xy(slot, h):
    a = np.radians(90.0 + h * 60.0 * slot)
    return np.array([R_HEX * np.cos(a), R_HEX * np.sin(a)])


# ════════════════════════════════════════════════════════════════════════════════
#  STAGE 1 -- CALIBRATION
# ════════════════════════════════════════════════════════════════════════════════
def turntable_mask(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 90) & (hsv[:, :, 1] < 48)).astype(np.uint8) * 255
    mask[:int(H * 0.60), :] = 0
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_corners(img):
    """Six silhouette corners of the hexagonal prism (sub-pixel, via line-fit
    intersections). A=left side vertex, B=near (lowest) vertex, C=right side
    vertex, D=far-right top vertex."""
    mask = turntable_mask(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    N = len(c)
    anchors = dict(Atop=(84, 921), Abot=(84, 1095), B=(224, 1197),
                   Cbot=(572, 1184), Ctop=(572, 977), D=(648, 910),
                   TL=(238, 873), TR=(501, 878))
    idx = {k: int(np.argmin(np.linalg.norm(c - np.array(v), axis=1)))
           for k, v in anchors.items()}

    def seg(i0, i1, trim=0.25):
        if i1 < i0:
            i1 += N
        p = c[[i % N for i in range(i0, i1 + 1)]]
        n = len(p); t = max(2, int(n * trim))
        return p[t:n - t] if n > 2 * t + 3 else p

    def line(a, b):
        vx, vy, x0, y0 = cv2.fitLine(seg(idx[a], idx[b]).astype(np.float32),
                                     cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array([vy, -vx, -(vy * x0 - vx * y0)])

    def inter(l1, l2):
        a1, b1, c1 = l1; a2, b2, c2 = l2; d = a1 * b2 - a2 * b1
        return np.array([(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d])

    Lv, BL, BR = line('Atop', 'Abot'), line('Abot', 'B'), line('B', 'Cbot')
    Rv, TR, TL = line('Cbot', 'Ctop'), line('Ctop', 'D'), line('TL', 'Atop')
    return {'Atop': inter(Lv, TL), 'Abot': inter(Lv, BL), 'B': inter(BL, BR),
            'Cbot': inter(BR, Rv), 'Ctop': inter(Rv, TR), 'D': inter(TR, line('D', 'TR'))}


def calibrate_pose(corners):
    """solvePnP + focal refinement (A,B,C,D = consecutive hexagon vertices)."""
    names = ['Atop', 'Abot', 'B', 'Cbot', 'Ctop', 'D']
    ip = np.array([corners[k] for k in names], float)
    best = None
    for h in (+1, -1):
        A, B, C, D = (_hex_xy(0, h), _hex_xy(1, h), _hex_xy(2, h), _hex_xy(3, h))
        obj = np.array([[*A, Z_TOP], [*A, Z_BOT], [*B, Z_BOT],
                        [*C, Z_BOT], [*C, Z_TOP], [*D, Z_TOP]], float)
        K0 = np.array([[1280., 0, CX], [0, 1280., CY], [0, 0, 1.]])
        _, rv0, tv0 = cv2.solvePnP(obj, ip, K0, None, flags=cv2.SOLVEPNP_ITERATIVE)

        def resid(x):
            K = np.array([[x[6], 0, CX], [0, x[6], CY], [0, 0, 1.]])
            pr, _ = cv2.projectPoints(obj, x[:3].reshape(3, 1),
                                      x[3:6].reshape(3, 1), K, None)
            return (pr.reshape(-1, 2) - ip).ravel()

        sol = least_squares(resid, np.hstack([rv0.ravel(), tv0.ravel(), 1280.]),
                            method='lm', max_nfev=300)
        rms = np.sqrt((resid(sol.x) ** 2).mean())
        if best is None or rms < best[0]:
            best = (rms, h, sol.x, obj, ip)
    rms, h, x, obj, ip = best
    f = x[6]
    K = np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.]])
    rvec, tvec = x[:3].reshape(3, 1), x[3:6].reshape(3, 1)
    Rm, _ = cv2.Rodrigues(rvec)
    pr, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    errs = np.linalg.norm(pr.reshape(-1, 2) - ip, axis=1)
    return dict(K=K, f=f, h=h, rvec=rvec, tvec=tvec, R=Rm, t=tvec.ravel(),
                C=(-Rm.T @ tvec).ravel(), rms=rms, errs=errs, names=names)


def fit_laser_plane(cal, n_frames=20):
    """Laser plane  a*x + b*y = 0  (contains the z axis), fit from the stripe on
    the bare turntable top face (z=Z_TOP)."""
    K, R, C, Kinv, h = cal['K'], cal['R'], cal['C'], np.linalg.inv(cal['K']), cal['h']
    top = np.array([[*_hex_xy(k, h), Z_TOP] for k in range(6)], float)
    poly = cv2.projectPoints(top, cal['rvec'], cal['tvec'], K, None)[0].reshape(-1, 2).astype(np.float32)
    pts3d = []
    for i in range(n_frames):
        wo = read_image(os.path.join(WITHOUT, f'{i:04d}.jpg'))
        wl = read_image(os.path.join(WITHLASER, f'{i:04d}.jpg'))
        # plane fit uses G-max(R,B) on the bright WHITE table (clean there); the
        # object stripe (incl. the red hat) uses excess-green 2G-R-B elsewhere.
        d = wl.astype(np.float32) - wo.astype(np.float32)
        resp = d[:, :, 1] - np.maximum(d[:, :, 2], d[:, :, 0])
        hsv = cv2.cvtColor(wo, cv2.COLOR_BGR2HSV)
        table = (hsv[:, :, 2] > 110) & (hsv[:, :, 1] < 48)
        H, W = resp.shape
        for v in range(int(H * 0.60), H):
            u = int(np.argmax(resp[v]))
            if resp[v, u] < PLANE_THRESH:
                continue
            if cv2.pointPolygonTest(poly, (float(u), float(v)), False) < 0:
                continue
            if not table[v, max(0, u - 2):u + 3].any():
                continue
            lo, hi = max(0, u - 4), min(W, u + 5)
            w = np.clip(resp[v, lo:hi], 0, None); s = w.sum()
            if s < 1e-6:
                continue
            us = float((w * np.arange(lo, hi)).sum() / s)
            ray = R.T @ (Kinv @ np.array([us, v, 1.0]))
            pts3d.append(C + (Z_TOP - C[2]) / ray[2] * ray)
    xy = np.array(pts3d)[:, :2]
    _, V = np.linalg.eigh(xy.T @ xy)          # line through origin = smallest evec
    resid = np.sqrt(((xy @ V[:, 0]) ** 2).mean())
    return V[:, 0], resid, len(pts3d)


def draw_overlay(img, cal, plane):
    K, h = cal['K'], cal['h']
    allv = np.array([[*_hex_xy(k, h), z] for k in range(6) for z in (Z_TOP, Z_BOT)])
    pr = cv2.projectPoints(allv, cal['rvec'], cal['tvec'], K, None)[0].reshape(-1, 2)
    vis = img.copy()
    for k in range(6):
        t1, t2 = 2 * k, 2 * ((k + 1) % 6)
        cv2.line(vis, tuple(pr[t1].astype(int)), tuple(pr[t2].astype(int)), (0, 255, 255), 2)
        cv2.line(vis, tuple(pr[t1 + 1].astype(int)), tuple(pr[t2 + 1].astype(int)), (255, 128, 0), 2)
        cv2.line(vis, tuple(pr[t1].astype(int)), tuple(pr[t1 + 1].astype(int)), (0, 255, 0), 2)
    a, b = plane; d = np.array([b, -a])
    for z, col in ((Z_TOP, (0, 0, 255)), (Z_BOT, (0, 0, 200))):
        q = cv2.projectPoints(np.array([[*(30 * d), z], [*(-30 * d), z]]),
                              cal['rvec'], cal['tvec'], K, None)[0].reshape(-1, 2).astype(int)
        cv2.line(vis, tuple(q[0]), tuple(q[1]), col, 2)
    return vis


def calibrate():
    print('[1/3] CALIBRATION')
    img0 = read_image(os.path.join(WITHOUT, '0000.jpg'))
    cal = calibrate_pose(detect_corners(img0))
    cal['plane'], resid, npts = fit_laser_plane(cal)
    print(f'  focal f = {cal["f"]:.1f} px   reproj RMS = {cal["rms"]:.3f} px '
          f'(max {cal["errs"].max():.3f})')
    print(f'  camera centre = ({cal["C"][0]:.1f}, {cal["C"][1]:.1f}, {cal["C"][2]:.1f}) mm')
    print(f'  laser plane  {cal["plane"][0]:+.4f} x {cal["plane"][1]:+.4f} y = 0  '
          f'(residual {resid:.3f} mm, {npts} pts)')
    cv2.imencode('.png', draw_overlay(read_image(os.path.join(WITHLASER, '0000.jpg')),
                 cal, cal['plane']))[1].tofile(os.path.join(HERE, 'calib_overlay.png'))
    np.savez(os.path.join(HERE, 'calib.npz'), K=cal['K'], R=cal['R'], t=cal['t'],
             rvec=cal['rvec'], tvec=cal['tvec'], C=cal['C'], f=cal['f'], h=cal['h'],
             laser_plane=cal['plane'], Z_TOP=Z_TOP, Z_BOT=Z_BOT, R_HEX=R_HEX)
    return cal


# ════════════════════════════════════════════════════════════════════════════════
#  STAGE 2 -- RECONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════════
def extract_stripe(with_l, without_l):
    """(M,2) array of [u_subpixel, v] laser-stripe points, one per image row."""
    resp = laser_response(with_l, without_l)
    H, W = resp.shape
    peak_col = np.argmax(resp, axis=1)
    peak_val = resp[np.arange(H), peak_col]
    pts = []
    for v in np.where(peak_val > LASER_THRESH)[0]:
        u = peak_col[v]
        lo, hi = max(0, u - HALF_WIN), min(W, u + HALF_WIN + 1)
        w = np.clip(resp[v, lo:hi], 0, None); s = w.sum()
        if s < 1e-6:
            continue
        pts.append((float((w * np.arange(lo, hi)).sum() / s), float(v)))
    return np.array(pts) if pts else np.empty((0, 2))


def triangulate(uv, K, R, C, plane):
    """uv:(M,2) pixels -> (M,3) world points on the plane a*x+b*y=0."""
    a, b = plane
    rays = (R.T @ (np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T)).T
    denom = a * rays[:, 0] + b * rays[:, 1]
    good = np.abs(denom) > 1e-9
    s = np.full(len(uv), np.nan); s[good] = -(a * C[0] + b * C[1]) / denom[good]
    return C[None, :] + s[:, None] * rays, good & (s > 0)


def bilinear(img, uv):
    H, W = img.shape[:2]
    u = np.clip(uv[:, 0], 0, W - 1.001); v = np.clip(uv[:, 1], 0, H - 1.001)
    x0 = u.astype(int); y0 = v.astype(int); x1 = x0 + 1; y1 = y0 + 1
    dx = (u - x0)[:, None]; dy = (v - y0)[:, None]
    c = (img[y0, x0] * (1 - dx) * (1 - dy) + img[y0, x1] * dx * (1 - dy)
         + img[y1, x0] * (1 - dx) * dy + img[y1, x1] * dx * dy)
    return c[:, ::-1]  # BGR -> RGB


def voxel_downsample(pts, cols, leaf):
    _, idx = np.unique(np.floor(pts / leaf).astype(np.int64), axis=0, return_index=True)
    return pts[idx], cols[idx]


def knn_filter(pts, cols, k=KNN_K, sigma=KNN_SIGMA):
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    md = d[:, 1:].mean(axis=1)
    med = np.median(md); mad = np.median(np.abs(md - med)) + 1e-9
    keep = md < med + sigma * 1.4826 * mad
    return pts[keep], cols[keep]


def radius_outlier(pts, cols, radius=ROR_RADIUS, min_pts=ROR_MIN):
    cnt = cKDTree(pts).query_ball_point(pts, radius, return_length=True)
    keep = cnt >= min_pts
    return pts[keep], cols[keep]


def reconstruct(cal):
    print('[2/3] RECONSTRUCTION')
    K, R, C, plane = cal['K'], cal['R'], cal['C'], cal['plane']
    all_pts, all_col = [], []
    for i in range(N_FRAMES):
        wl = read_image(os.path.join(WITHLASER, f'{i:04d}.jpg'))
        wo = read_image(os.path.join(WITHOUT,  f'{i:04d}.jpg'))
        uv = extract_stripe(wl, wo)
        if len(uv) == 0:
            continue
        X, ok = triangulate(uv, K, R, C, plane)
        X, uv = X[ok], uv[ok]
        r = np.hypot(X[:, 0], X[:, 1])
        m = (r <= R_MAX) & (X[:, 2] >= Z_BOT - Z_PAD) & (X[:, 2] <= Z_MAX)
        X, uv = X[m], uv[m]
        if len(X) == 0:
            continue
        rgb = bilinear(wo, uv)
        X = (rot_z(-ROT_SIGN * DEG_PER_FRAME * i) @ X.T).T   # -> frame-0 object frame
        all_pts.append(X); all_col.append(rgb)
    pts = np.vstack(all_pts); cols = np.vstack(all_col)
    print(f'  raw {len(pts)} pts', end='')
    pts, cols = voxel_downsample(pts, cols, VOXEL)
    pts, cols = knn_filter(pts, cols)
    pts, cols = radius_outlier(pts, cols)
    cols = np.clip(cols, 0, 255).astype(np.uint8)
    print(f'  ->  {len(pts)} after voxel+outlier removal')
    np.savetxt(os.path.join(HERE, f'{STUDENT_ID}.xyz'), np.c_[pts, cols],
               fmt='%.4f %.4f %.4f %d %d %d')
    write_ply(os.path.join(HERE, f'{STUDENT_ID}.ply'), pts, cols)
    print(f'  saved {STUDENT_ID}.xyz and {STUDENT_ID}.ply')
    return pts


# ════════════════════════════════════════════════════════════════════════════════
#  STAGE 3 -- EVALUATION  (reads GroundTruth.ply; --no-eval to skip)
# ════════════════════════════════════════════════════════════════════════════════
def load_gt_mesh(path):
    import re
    data = open(path, 'rb').read()
    he = data.find(b'end_header\n') + len(b'end_header\n')
    nv = int(re.search(rb'element vertex (\d+)', data).group(1))
    nf = int(re.search(rb'element face (\d+)', data).group(1))
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])
    V = np.frombuffer(data, dtype=dt, count=nv, offset=he)
    verts = np.stack([V['x'], V['y'], V['z']], 1).astype(np.float64)
    buf = np.frombuffer(data, dtype=np.uint8, offset=he + nv * 16)
    tris = []; p = 0
    for _ in range(nf):
        n = buf[p]; ids = buf[p + 1:p + 1 + 4 * n].view('<u4')
        for k in range(1, n - 1):
            tris.append((ids[0], ids[k], ids[k + 1]))
        p += 1 + 4 * n
    return verts, np.array(tris, dtype=np.int64)


def sample_mesh(verts, faces, n=400000):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    pick = np.random.choice(len(faces), size=n, p=area / area.sum())
    u = np.random.rand(n, 1); w = np.random.rand(n, 1)
    over = (u + w > 1).ravel(); u[over] = 1 - u[over]; w[over] = 1 - w[over]
    a, b, c = v0[pick], v1[pick], v2[pick]
    return a + u * (b - a) + w * (c - a)


def icp(src, tree, iters=12):
    P = src.copy()
    for _ in range(iters):
        _, idx = tree.query(P); Q = tree.data[idx]
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1; Rm = Vt.T @ U.T
        P = (Rm @ P.T).T + (mq - Rm @ mp)
    return P


def evaluate(rec):
    print('[3/3] EVALUATION vs GroundTruth.ply (evaluation only)')
    np.random.seed(0)
    verts, faces = load_gt_mesh(GT_PLY)
    gt = sample_mesh(verts, faces); tree_gt = cKDTree(gt)
    # align: yaw search (+mirror check) then rigid ICP -- diagnostic only
    best = None
    sub = rec[np.random.choice(len(rec), min(8000, len(rec)), replace=False)]
    for mirror in (1, -1):
        S = sub.copy(); S[:, 0] *= mirror
        for yaw in range(0, 360, 2):
            d, _ = tree_gt.query((rot_z(yaw) @ S.T).T)
            if best is None or d.mean() < best[0]:
                best = (d.mean(), yaw, mirror)
    _, yaw, mirror = best
    rec_a = rec.copy(); rec_a[:, 0] *= mirror
    rec_a = icp((rot_z(yaw) @ rec_a.T).T, tree_gt)
    d_r2g, _ = tree_gt.query(rec_a)
    d_g2r, _ = cKDTree(rec_a).query(gt)

    def stats(d, tag):
        print(f'  {tag:9s} mean={d.mean():.3f} median={np.median(d):.3f} '
              f'RMS={np.sqrt((d**2).mean()):.3f} 95%={np.percentile(d,95):.3f} '
              f'max={d.max():.3f} mm')
    print(f'  aligned: mirror={mirror} yaw={yaw} deg')
    stats(d_r2g, 'recon->GT'); stats(d_g2r, 'GT->recon')
    print(f'  coverage (GT within 1mm of recon): {(d_g2r < 1.0).mean()*100:.1f} %')
    # coloured error PLY + histogram
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    t = np.clip(d_r2g / max(np.percentile(d_r2g, 98), 1e-6), 0, 1)
    write_ply(os.path.join(HERE, 'error_map.ply'), rec_a,
              (plt.get_cmap('jet')(t)[:, :3] * 255).astype(np.uint8))
    plt.figure(figsize=(7, 4))
    plt.hist(d_r2g, bins=120, range=(0, np.percentile(d_r2g, 99)), color='#3070b0')
    plt.axvline(np.median(d_r2g), color='r', ls='--',
                label=f'median={np.median(d_r2g):.3f} mm')
    plt.xlabel('recon -> GT distance (mm)'); plt.ylabel('count')
    plt.title('Reconstruction accuracy vs GroundTruth'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'error_hist.png'), dpi=110)
    print('  saved error_map.ply and error_hist.png')


# ════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--no-eval', action='store_true',
                    help='skip the ground-truth comparison (Stage 3)')
    args = ap.parse_args()
    cal = calibrate()
    rec = reconstruct(cal)
    if not args.no_eval and os.path.exists(GT_PLY):
        evaluate(rec)
    elif not args.no_eval:
        print('GroundTruth.ply not found -- skipping evaluation.')


if __name__ == '__main__':
    main()
