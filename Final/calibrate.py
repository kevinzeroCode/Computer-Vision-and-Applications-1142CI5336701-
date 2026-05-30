"""
calibrate.py
Recover the camera extrinsics and the laser-plane equation for the slit-laser
turntable scanner, in the world frame whose origin is the turntable centre,
+Z = rotation axis (up).

Pipeline
--------
1. Detect the hexagonal-prism silhouette in withoutLaser/0000.jpg (bright, low
   saturation) and extract its 6 corners by intersecting line-fits of the
   silhouette edges (sub-pixel, robust to morphology rounding).
2. Match the 6 corners to the known turntable model (regular hexagon, circum-
   radius 25 mm, top face z=+1, bottom z=-14 -- read from GroundTruth.ply) and
   solve the pose with cv2.solvePnP, refining the focal length together with the
   pose (least-squares).  NOTE: the provided K lists f=720, but that value is
   inconsistent with the imaged 50 mm turntable; PnP against the known hexagon
   yields f~=1280 px (reproj RMS ~0.1 px), which we adopt.  cx=360, cy=640 are
   the image centre and check out exactly.
3. Fit the laser plane a*x + b*y = 0 (it contains the z rotation axis) from the
   laser stripe that falls on the bare turntable top face (z=+1).

Outputs: calib.npz  (K, R, t, rvec, tvec, laser_plane=[a,b], frame-0 corners)
         _calib_overlay.png  (visual check)
"""

import numpy as np
import cv2
import os
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
WITHLASER = os.path.join(HERE, 'ScandatawithLaser')
WITHOUT   = os.path.join(WITHLASER, 'withoutLaser')

# ─── Turntable model (mm), from GroundTruth.ply ─────────────────────────────────
R_HEX = 25.0          # circumradius (vertex-to-vertex diameter = 50 mm)
Z_TOP = 1.0           # top face
Z_BOT = -14.0         # bottom face  (thickness 15 mm)
CX, CY = 360.0, 640.0 # principal point (= image centre, confirmed)


# ─── Image I/O (Unicode-safe for the 台科 path) ─────────────────────────────────
def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ─── Turntable silhouette ───────────────────────────────────────────────────────
def turntable_mask(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((val > 90) & (sat < 48)).astype(np.uint8) * 255
    mask[:int(H * 0.60), :] = 0                       # table lives in lower frame
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _fit_line(pts):
    """Return homogeneous line [a,b,c] (a*x+b*y+c=0) through pts (Huber)."""
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER,
                                 0, 0.01, 0.01).ravel()
    return np.array([vy, -vx, -(vy * x0 - vx * y0)])


def _intersect(l1, l2):
    a1, b1, c1 = l1; a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    return np.array([(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d])


def detect_corners(img):
    """Six silhouette corners of the hexagonal prism (sub-pixel).

    Returns dict with keys Atop,Abot,B,Cbot,Ctop,D.
      A = left side vertex (vertical edge), B = near (lowest) vertex,
      C = right side vertex (vertical edge), D = far-right top vertex.
    """
    mask = turntable_mask(img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    N = len(c)

    # Rough anchors for frame 0 (fixed input); refined below by line intersection.
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
        return _fit_line(seg(idx[a], idx[b]))

    Lvert = line('Atop', 'Abot')      # left vertical edge
    botL  = line('Abot', 'B')         # bottom-left edge
    botR  = line('B', 'Cbot')         # bottom-right edge
    Rvert = line('Cbot', 'Ctop')      # right vertical edge
    topR  = line('Ctop', 'D')         # top-right edge
    topL  = line('TL', 'Atop')        # top-left edge
    topD  = line('D', 'TR')           # top edge past D

    return {
        'Atop': _intersect(Lvert, topL),
        'Abot': _intersect(Lvert, botL),
        'B':    _intersect(botL, botR),
        'Cbot': _intersect(botR, Rvert),
        'Ctop': _intersect(Rvert, topR),
        'D':    _intersect(topR, topD),
    }


# ─── Pose + focal calibration ───────────────────────────────────────────────────
def _hex_xy(slot, h):
    a = np.radians(90.0 + h * 60.0 * slot)
    return np.array([R_HEX * np.cos(a), R_HEX * np.sin(a)])


def calibrate_pose(corners):
    """solvePnP + focal refinement.  A,B,C,D are consecutive hexagon vertices.

    The world-frame azimuth (gauge) is fixed by placing vertex A at 90 deg; the
    turntable's true frame-0 azimuth is absorbed here and only matters as a
    global yaw vs the ground truth (handled in evaluation, never in recon)."""
    names = ['Atop', 'Abot', 'B', 'Cbot', 'Ctop', 'D']
    ip = np.array([corners[k] for k in names], float)

    best = None
    for h in (+1, -1):
        A, B, C, D = (_hex_xy(0, h), _hex_xy(1, h), _hex_xy(2, h), _hex_xy(3, h))
        obj = np.array([[*A, Z_TOP], [*A, Z_BOT], [*B, Z_BOT],
                        [*C, Z_BOT], [*C, Z_TOP], [*D, Z_TOP]], float)
        K0 = np.array([[1280., 0, CX], [0, 1280., CY], [0, 0, 1.]])
        ok, rv0, tv0 = cv2.solvePnP(obj, ip, K0, None,
                                    flags=cv2.SOLVEPNP_ITERATIVE)

        def resid(x):
            f = x[6]
            K = np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.]])
            pr, _ = cv2.projectPoints(obj, x[:3].reshape(3, 1),
                                      x[3:6].reshape(3, 1), K, None)
            return (pr.reshape(-1, 2) - ip).ravel()

        sol = least_squares(resid, np.hstack([rv0.ravel(), tv0.ravel(), 1280.]),
                            method='lm', max_nfev=300)
        rms = np.sqrt((resid(sol.x) ** 2).mean())
        if best is None or rms < best[0]:
            best = (rms, h, sol.x, obj, ip, names)

    rms, h, x, obj, ip, names = best
    f = x[6]
    K = np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.]])
    rvec, tvec = x[:3].reshape(3, 1), x[3:6].reshape(3, 1)
    Rm, _ = cv2.Rodrigues(rvec)
    C = (-Rm.T @ tvec).ravel()
    pr, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    errs = np.linalg.norm(pr.reshape(-1, 2) - ip, axis=1)
    return dict(K=K, f=f, h=h, rvec=rvec, tvec=tvec, R=Rm, t=tvec.ravel(),
                C=C, rms=rms, errs=errs, names=names, obj=obj, ip=ip)


# ─── Laser-plane fit ────────────────────────────────────────────────────────────
def laser_response(with_l, without_l):
    """Green-stripe response  G - max(R,B)  on the (laser - background) image."""
    diff = with_l.astype(np.float32) - without_l.astype(np.float32)
    return diff[:, :, 1] - np.maximum(diff[:, :, 2], diff[:, :, 0])   # BGR


def fit_laser_plane(cal, n_frames=20, thresh=40.0):
    """Laser plane  a*x + b*y = 0  (contains the z axis).

    Collect the laser stripe lying on the BARE turntable top face (z=Z_TOP) over
    the first n_frames, back-project each pixel to z=Z_TOP, and fit a line through
    the origin (= the plane's intersection with z=Z_TOP)."""
    K, R, C = cal['K'], cal['R'], cal['C']
    Kinv = np.linalg.inv(K)
    h = cal['h']
    top = np.array([[*_hex_xy(k, h), Z_TOP] for k in range(6)], float)
    pr, _ = cv2.projectPoints(top, cal['rvec'], cal['tvec'], K, None)
    poly = pr.reshape(-1, 2).astype(np.float32)

    pts3d = []
    for i in range(n_frames):
        wl = read_image(os.path.join(WITHLASER, f'{i:04d}.jpg'))
        wo = read_image(os.path.join(WITHOUT,  f'{i:04d}.jpg'))
        resp = laser_response(wl, wo)
        hsv = cv2.cvtColor(wo, cv2.COLOR_BGR2HSV)
        table = (hsv[:, :, 2] > 110) & (hsv[:, :, 1] < 48)
        H, W = resp.shape
        for v in range(int(H * 0.60), H):
            row = resp[v]
            u = int(np.argmax(row))
            if row[u] < thresh:
                continue
            if cv2.pointPolygonTest(poly, (float(u), float(v)), False) < 0:
                continue
            if not table[v, max(0, u - 2):u + 3].any():
                continue
            lo, hi = max(0, u - 4), min(W, u + 5)
            w = np.clip(row[lo:hi], 0, None); s = w.sum()
            if s < 1e-6:
                continue
            us = float((w * np.arange(lo, hi)).sum() / s)
            ray = R.T @ (Kinv @ np.array([us, v, 1.0]))
            t = (Z_TOP - C[2]) / ray[2]
            pts3d.append(C + t * ray)
    pts3d = np.array(pts3d)
    xy = pts3d[:, :2]
    # line through origin: normal = eigvec of smallest eigenvalue of xy^T xy
    M = xy.T @ xy
    w, V = np.linalg.eigh(M)
    a, b = V[:, 0]
    resid = np.sqrt(((xy @ V[:, 0]) ** 2).mean())
    return np.array([a, b]), resid, len(pts3d)


def draw_overlay(img, cal, plane):
    K, h = cal['K'], cal['h']
    allv = np.array([[*_hex_xy(k, h), z] for k in range(6) for z in (Z_TOP, Z_BOT)])
    pr, _ = cv2.projectPoints(allv, cal['rvec'], cal['tvec'], K, None)
    pr = pr.reshape(-1, 2)
    vis = img.copy()
    for k in range(6):
        t1, t2 = 2 * k, 2 * ((k + 1) % 6)
        cv2.line(vis, tuple(pr[t1].astype(int)), tuple(pr[t2].astype(int)), (0, 255, 255), 2)
        cv2.line(vis, tuple(pr[t1 + 1].astype(int)), tuple(pr[t2 + 1].astype(int)), (255, 128, 0), 2)
        cv2.line(vis, tuple(pr[t1].astype(int)), tuple(pr[t1 + 1].astype(int)), (0, 255, 0), 2)
    # laser plane: draw its line on z=Z_TOP and z=Z_BOT (dir perpendicular to (a,b))
    a, b = plane; d = np.array([b, -a])
    for z, col in ((Z_TOP, (0, 0, 255)), (Z_BOT, (0, 0, 200))):
        p1 = np.array([*( 30 * d), z]); p2 = np.array([*(-30 * d), z])
        q, _ = cv2.projectPoints(np.array([p1, p2]), cal['rvec'], cal['tvec'], K, None)
        q = q.reshape(-1, 2).astype(int)
        cv2.line(vis, tuple(q[0]), tuple(q[1]), col, 2)
    return vis


def main():
    img0 = read_image(os.path.join(WITHOUT, '0000.jpg'))
    corners = detect_corners(img0)
    print('Detected corners (px):')
    for k, v in corners.items():
        print(f'  {k:5s} ({v[0]:7.2f}, {v[1]:7.2f})')

    cal = calibrate_pose(corners)
    print(f'\nFocal f = {cal["f"]:.1f} px  (cx={CX}, cy={CY})')
    print(f'Reprojection RMS = {cal["rms"]:.3f} px   max = {cal["errs"].max():.3f} px')
    print('Per-corner err:', dict(zip(cal['names'], np.round(cal['errs'], 2))))
    print(f'Camera centre  = ({cal["C"][0]:.1f}, {cal["C"][1]:.1f}, {cal["C"][2]:.1f}) mm'
          f'   |C| = {np.linalg.norm(cal["C"]):.1f} mm')
    az = np.degrees(np.arctan2(cal['C'][1], cal['C'][0]))
    el = np.degrees(np.arctan2(cal['C'][2], np.hypot(cal['C'][0], cal['C'][1])))
    print(f'Camera azimuth = {az:.1f} deg   elevation = {el:.1f} deg')

    plane, resid, npts = fit_laser_plane(cal)
    a, b = plane
    print(f'\nLaser plane:  {a:+.4f} x {b:+.4f} y = 0   ({npts} top-face pts, '
          f'residual RMS = {resid:.3f} mm)')

    img_with = read_image(os.path.join(WITHLASER, '0000.jpg'))
    vis = draw_overlay(img_with, cal, plane)
    cv2.imencode('.png', vis)[1].tofile(os.path.join(HERE, 'calib_overlay.png'))

    np.savez(os.path.join(HERE, 'calib.npz'),
             K=cal['K'], R=cal['R'], t=cal['t'], rvec=cal['rvec'],
             tvec=cal['tvec'], C=cal['C'], f=cal['f'], h=cal['h'],
             laser_plane=plane, Z_TOP=Z_TOP, Z_BOT=Z_BOT, R_HEX=R_HEX,
             corners=np.array([corners[k] for k in cal['names']]),
             names=np.array(cal['names']))
    print('Saved calib.npz and _calib_overlay.png')


if __name__ == '__main__':
    main()
