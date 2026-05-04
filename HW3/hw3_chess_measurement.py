"""
HW3: Chess Piece Distance and Height Estimation
Student: M11415015

Estimates the Euclidean distance from the camera optical center to the chess
piece base, and the physical height of the chess piece, using perspective
projection from a single image with a known checkerboard (1 cm/square).

World coordinate frame (cm):
  Origin = axis marker in image (lower-left of checkerboard)
  +X     = red arrow direction  (rightward along board)
  +Y     = green arrow direction (into depth / up the image)
  +Z     = upward, perpendicular to board (right-hand rule)

Usage:
  python hw3_chess_measurement.py          # run measurement
  python hw3_chess_measurement.py --pick   # interactive point picker
"""

import sys
import cv2
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_PATH  = "ChessonChecker.png"
OUTPUT_PATH = "result.jpg"

# ─── CAMERA INTRINSICS ───────────────────────────────────────────────────────
K = np.array([
    [1866.7,    0.0,  960.0],
    [   0.0, 1866.7,  540.0],
    [   0.0,    0.0,    1.0],
], dtype=np.float64)
DIST = np.zeros((4, 1), dtype=np.float64)   # rendered scene → no lens distortion

# ─── MANUALLY SELECTED CORRESPONDENCE POINTS ─────────────────────────────────
# World coordinate frame (Z=0 plane = board surface):
#   Origin (0,0,0): corner at pixel (396, 751) — verified by homography fit
#   +X axis: goes rightward along board near-edge (→ right+slightly-up in image)
#   +Y axis: goes into scene depth (→ left+up in image, toward far board)
#   Unit: 1.0 = 1 cm (one checkerboard square)
#
# Points 0–3: column X=0 (leftmost visible column, detected by Harris+subPix)
# Points 4  : column X=1 (detected by Harris+subPix, exact homography match)
# Points 5–11: columns X=2..5 (ground-truth corners from homography fit)

pts_3d = np.array([
    [ 0.0,  0.0,  0.0],   # 0  world origin
    [ 0.0,  2.0,  0.0],   # 1  (0,2)
    [ 0.0,  4.0,  0.0],   # 2  (0,4)
    [ 0.0,  6.0,  0.0],   # 3  (0,6)
    [ 1.0,  0.0,  0.0],   # 4  (1,0)
    [ 2.0,  0.0,  0.0],   # 5  (2,0)
    [ 2.0,  3.0,  0.0],   # 6  (2,3)
    [ 3.0,  0.0,  0.0],   # 7  (3,0)
    [ 3.0,  3.0,  0.0],   # 8  (3,3)
    [ 4.0,  2.0,  0.0],   # 9  (4,2)
    [ 5.0,  0.0,  0.0],   # 10 (5,0)
    [ 5.0,  4.0,  0.0],   # 11 (5,4)
], dtype=np.float64)

# Pixel coordinates confirmed by Harris corner detection + cornerSubPix refinement.
# Points 5–11 are ground-truth (< 0.15 px homography residual).
# Points 0–4 verified by matching Harris-detected corners to homography predictions.
pts_2d = np.array([
    [[ 396.4,  750.5]],   # 0  (0,0) — Harris corner, exact homography match
    [[ 340.7,  639.9]],   # 1  (0,2) — Harris corner
    [[ 294.9,  548.9]],   # 2  (0,4) — Harris corner
    [[ 256.5,  472.7]],   # 3  (0,6) — Harris corner
    [[ 521.3,  718.3]],   # 4  (1,0) — Harris corner, exact homography match
    [[ 639.3,  687.4]],   # 5  (2,0) — ground truth
    [[ 531.3,  545.3]],   # 6  (2,3) — ground truth
    [[ 750.6,  658.5]],   # 7  (3,0) — ground truth
    [[ 631.0,  523.3]],   # 8  (3,3) — ground truth
    [[ 765.7,  541.5]],   # 9  (4,2) — ground truth
    [[ 956.6,  604.8]],   # 10 (5,0) — ground truth
    [[ 777.7,  448.1]],   # 11 (5,4) — ground truth
], dtype=np.float64)

# Chess piece pixels — (u, v) = (col, row)
#   base: lowest detected gold pixel at board contact (HSV gold mask, shadow excluded)
#   top : topmost gold pixel (tip of crown cross)
BASE_PX = np.array([1167.0, 743.0])
TOP_PX  = np.array([1179.0, 387.0])

# ─── QUALITY THRESHOLDS ──────────────────────────────────────────────────────
REPROJ_WARN  = 2.0    # px — target
REPROJ_HARD  = 5.0    # px — hard stop if exceeded
MIN_INLIERS  = 6
DENOM_THRESH = 1e-4   # near-zero denominator guard for height formula
HEIGHT_AGREE = 0.5    # cm — max disagreement between u/v height solutions

# =============================================================================
# UTILITIES
# =============================================================================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if img.ndim == 2:                          # grayscale → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                      # RGBA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# =============================================================================
# INTERACTIVE POINT PICKER (run with --pick)
# =============================================================================

def pick_points(img_path: str) -> None:
    """Open image in a window; click to record (u,v) coordinates."""
    img = load_image(img_path)
    clicks: list[tuple[int, int]] = []
    display = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            n = len(clicks)
            clicks.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display, str(n), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Pick Points — press Q when done", display)
            print(f"  [{n}]  u={x}, v={y}")

    win = "Pick Points — press Q when done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_click)
    cv2.imshow(win, display)

    print("\nClick each checkerboard corner in ORDER matching pts_3d.")
    print("Press Q or Escape when done.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
    cv2.destroyAllWindows()

    print("\n=== Paste these into pts_2d ===")
    for i, (x, y) in enumerate(clicks):
        print(f"    [[{x:4d}, {y:4d}]],   # {i}")


# =============================================================================
# STEP 1: CAMERA POSE ESTIMATION
# =============================================================================

def estimate_pose(pts_3d: np.ndarray, pts_2d: np.ndarray):
    """
    Estimate camera extrinsics using solvePnPRansac (robust to outlier clicks),
    then refine with iterative PnP on inliers only.

    Returns R (3×3), t (3,), inlier_mask (N,), rvec, tvec.
    """
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.reshape(-1, 1, 3),
        pts_2d,
        K, DIST,
        iterationsCount=1000,
        reprojectionError=REPROJ_HARD,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ret:
        raise RuntimeError(
            "solvePnPRansac failed — increase points or check coordinate convention."
        )
    if inliers is None or len(inliers) < MIN_INLIERS:
        n = 0 if inliers is None else len(inliers)
        raise RuntimeError(
            f"Only {n} inliers (need ≥ {MIN_INLIERS}). Re-select reference points."
        )

    # Refine on inlier subset
    idx = inliers.flatten()
    ret2, rvec, tvec = cv2.solvePnP(
        pts_3d[idx].reshape(-1, 1, 3),
        pts_2d[idx],
        K, DIST,
        rvec, tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()

    print(f"\n[Step 1] Camera Pose")
    print(f"  Inliers used: {len(idx)} / {len(pts_3d)}")
    print(f"  R =\n{R}")
    print(f"  t = {t}  (cm)")

    return R, t, idx, rvec, tvec


# =============================================================================
# STEP 2: REPROJECTION ERROR CHECK
# =============================================================================

def check_reprojection(pts_3d: np.ndarray, pts_2d: np.ndarray,
                        inlier_idx: np.ndarray, rvec, tvec) -> float:
    pts_in_3d = pts_3d[inlier_idx].reshape(-1, 1, 3)
    pts_in_2d = pts_2d[inlier_idx]

    proj, _ = cv2.projectPoints(pts_in_3d, rvec, tvec, K, DIST)
    errors = np.linalg.norm(pts_in_2d.squeeze() - proj.squeeze(), axis=1)
    mean_err = float(errors.mean())
    max_err  = float(errors.max())

    print(f"\n[Step 2] Reprojection Error")
    for i, (idx, e) in enumerate(zip(inlier_idx, errors)):
        print(f"  pt{idx:2d}  {e:.2f} px")
    print(f"  Mean: {mean_err:.3f} px   Max: {max_err:.3f} px  (target < {REPROJ_WARN} px)")

    if mean_err > REPROJ_HARD:
        raise RuntimeError(
            f"Mean reprojection error {mean_err:.2f} px exceeds {REPROJ_HARD} px hard limit."
            "\nRe-select reference points more carefully."
        )
    if mean_err > REPROJ_WARN:
        print(f"  WARNING: error > {REPROJ_WARN} px — consider re-checking clicked pixels.")

    return mean_err


# =============================================================================
# STEP 3: BACK-PROJECT BASE PIXEL TO Z=0
# =============================================================================

def backproject_base(R: np.ndarray, t: np.ndarray,
                     u: float, v: float, camera_center: np.ndarray):
    """
    Two methods; chooses the more numerically stable one.

    Method A — Ground-plane homography:
      H = K [r1 | r2 | t];  [X,Y,1] = H^{-1} [u,v,1]

    Method B — Ray-plane intersection (Z=0 plane):
      ray direction d_world = R^T K^{-1} [u,v,1]
      lambda = -C_z / d_z;  P = C + lambda*d
    """
    # Method A
    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    H_g = K @ np.hstack([r1, r2, t.reshape(3, 1)])
    cond_A = np.linalg.cond(H_g)

    # Method B
    d_cam   = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d_world = R.T @ d_cam

    if cond_A < 1e6:
        wh = np.linalg.inv(H_g) @ np.array([u, v, 1.0])
        X_b, Y_b = wh[0] / wh[2], wh[1] / wh[2]
        method = "homography"
    else:
        if abs(d_world[2]) < 1e-10:
            raise RuntimeError("Ray is parallel to Z=0 plane — cannot back-project base.")
        lam = -camera_center[2] / d_world[2]
        P   = camera_center + lam * d_world
        X_b, Y_b = P[0], P[1]
        method = "ray-plane"

    print(f"\n[Step 3] Base Back-projection  (method: {method})")
    print(f"  base_px = ({u:.1f}, {v:.1f})")
    print(f"  world base = ({X_b:.4f}, {Y_b:.4f}, 0) cm")

    return float(X_b), float(Y_b)


# =============================================================================
# STEP 4: HEIGHT CALCULATION (dual-formula, averaged)
# =============================================================================

def compute_height(R: np.ndarray, t: np.ndarray,
                   X_b: float, Y_b: float,
                   u_top: float, v_top: float) -> float:
    """
    Solves for Z_top where P_top = (X_b, Y_b, Z_top) projects to (u_top, v_top).

    From s*[u;v;1] = K [R|t] [X_b; Y_b; Z_top; 1]:
      Let a = R[0,:2]·[X_b,Y_b] + t[0]   (camera x at Z=0)
          b = R[1,:2]·[X_b,Y_b] + t[1]   (camera y at Z=0)
          c = R[2,:2]·[X_b,Y_b] + t[2]   (camera z at Z=0)

    v-equation: Z_top = (fy*b - (v-cy)*c) / ((v-cy)*R22 - fy*R12)
    u-equation: Z_top = (fx*a - (u-cx)*c) / ((u-cx)*R22 - fx*R02)
    """
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    a = R[0, 0]*X_b + R[0, 1]*Y_b + t[0]
    b = R[1, 0]*X_b + R[1, 1]*Y_b + t[1]
    c = R[2, 0]*X_b + R[2, 1]*Y_b + t[2]

    denom_v = (v_top - cy)*R[2, 2] - fy*R[1, 2]
    denom_u = (u_top - cx)*R[2, 2] - fx*R[0, 2]

    v_ok = abs(denom_v) > DENOM_THRESH
    u_ok = abs(denom_u) > DENOM_THRESH

    Z_top_v = ((fy*b - (v_top - cy)*c) / denom_v) if v_ok else None
    Z_top_u = ((fx*a - (u_top - cx)*c) / denom_u) if u_ok else None

    print(f"\n[Step 4] Height Calculation")
    print(f"  a={a:.6f}  b={b:.6f}  c={c:.6f}")
    print(f"  Z_top_v = {Z_top_v:.4f} cm  (denom_v={denom_v:.6f})" if v_ok else "  Z_top_v: denominator too small, skipped")
    print(f"  Z_top_u = {Z_top_u:.4f} cm  (denom_u={denom_u:.6f})" if u_ok else "  Z_top_u: denominator too small, skipped")

    if v_ok and u_ok:
        diff = abs(Z_top_v - Z_top_u)
        if diff > HEIGHT_AGREE:
            print(f"  WARNING: |Z_v - Z_u| = {diff:.3f} cm > {HEIGHT_AGREE} cm — using v-equation only")
            print(f"  (u-eq unreliable: crown tip is horizontally offset from piece axis;"
                  f" denom_u={denom_u:.1f} vs denom_v={denom_v:.1f}; v-eq is {abs(denom_v/denom_u):.0f}x more stable)")
            height = Z_top_v
        else:
            height = (Z_top_v + Z_top_u) / 2.0
            print(f"  Both solutions agree (diff={diff:.4f} cm) → average = {height:.4f} cm")
    elif v_ok:
        height = Z_top_v
        print("  Using v-equation (u-equation denominator too small)")
    elif u_ok:
        height = Z_top_u
        print("  Using u-equation (v-equation denominator too small)")
    else:
        raise RuntimeError("Both height denominators near zero — geometry is degenerate.")

    if height <= 0:
        raise RuntimeError(f"Height = {height:.4f} cm ≤ 0 — check TOP_PX selection.")

    return float(height)


# =============================================================================
# STEP 5: DISTANCE CALCULATION
# =============================================================================

def compute_distance(camera_center: np.ndarray, X_b: float, Y_b: float) -> float:
    """Euclidean distance from camera optical center to chess piece base (cm)."""
    P_base = np.array([X_b, Y_b, 0.0])
    dist = float(np.linalg.norm(camera_center - P_base))

    print(f"\n[Step 5] Distance Calculation")
    print(f"  Camera center C = {camera_center}  (cm)")
    print(f"  Chess base     = ({X_b:.4f}, {Y_b:.4f}, 0)  (cm)")
    print(f"  ||C - base||   = {dist:.4f} cm")

    if dist <= 0:
        raise RuntimeError(f"Distance = {dist:.4f} cm is non-positive — check pose.")

    return dist


# =============================================================================
# STEP 6: VISUALIZATION
# =============================================================================

def visualize(img: np.ndarray,
              pts_2d_all:   np.ndarray,
              inlier_idx:   np.ndarray,
              rvec,
              tvec,
              base_px:      np.ndarray,
              top_px:       np.ndarray,
              distance:     float,
              height:       float,
              mean_err:     float,
              out_path:     str) -> None:
    """
    Annotate image for verification and grading:
      green circles  = original inlier reference points
      yellow circles = reprojected reference points (error visible at a glance)
      cyan           = chess piece base
      magenta        = chess piece crown
      white line     = height indicator
      colored arrows = world axes from origin
      text overlay   = results
    """
    out = img.copy()

    # ── Reprojected reference points (yellow) ──
    pts_in_3d = pts_3d[inlier_idx].reshape(-1, 1, 3)
    proj, _   = cv2.projectPoints(pts_in_3d, rvec, tvec, K, DIST)
    proj_sq   = proj.squeeze()

    for i, (idx, p) in enumerate(zip(inlier_idx, proj_sq)):
        pu, pv = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (pu, pv), 7, (0, 255, 255), 2)  # yellow

    # ── Original inlier reference points (green) ──
    for i, idx in enumerate(inlier_idx):
        u, v = int(round(pts_2d[idx, 0, 0])), int(round(pts_2d[idx, 0, 1]))
        cv2.circle(out, (u, v), 5, (0, 200, 0), -1)
        cv2.putText(out, str(idx), (u + 7, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # ── World axes from origin point ──
    origin_px = (int(round(pts_2d[0, 0, 0])), int(round(pts_2d[0, 0, 1])))

    def project_point(Xw, Yw, Zw):
        p3d = np.array([[[Xw, Yw, Zw]]], dtype=np.float64)
        pp, _ = cv2.projectPoints(p3d, rvec, tvec, K, DIST)
        return (int(round(pp[0, 0, 0])), int(round(pp[0, 0, 1])))

    tip_x = project_point(2.0, 0.0, 0.0)
    tip_y = project_point(0.0, 2.0, 0.0)
    tip_z = project_point(0.0, 0.0, 2.0)
    cv2.arrowedLine(out, origin_px, tip_x, (0, 0, 220), 2, tipLength=0.2)   # red   = +X
    cv2.arrowedLine(out, origin_px, tip_y, (0, 200, 0), 2, tipLength=0.2)   # green = +Y
    cv2.arrowedLine(out, origin_px, tip_z, (200, 0, 0), 2, tipLength=0.2)   # blue  = +Z

    # ── Chess piece markers ──
    bu, bv = int(round(base_px[0])), int(round(base_px[1]))
    tu, tv = int(round(top_px[0])),  int(round(top_px[1]))
    cv2.circle(out, (bu, bv), 8, (255, 200, 0), -1)          # cyan  = base
    cv2.circle(out, (tu, tv), 8, (255, 0, 200), -1)          # magenta = top
    cv2.line(out, (bu, bv), (tu, tv), (255, 255, 255), 2)    # white line = height
    cv2.putText(out, "Base", (bu + 10, bv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(out, "Top",  (tu + 10, tv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 2)

    # ── Text overlay ──
    lines = [
        f"Distance: {distance:.2f} cm",
        f"Height:   {height:.2f} cm",
        f"Reproj err (mean): {mean_err:.2f} px",
    ]
    x0, y0 = 20, 40
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (x0, y0 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),   3)
        cv2.putText(out, ln, (x0, y0 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    cv2.imwrite(out_path, out)
    print(f"\n[Step 6] Visualization saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("HW3: Chess Piece Distance and Height Estimation")
    print("=" * 60)

    img = load_image(IMAGE_PATH)

    # 1. Camera pose
    R, t, inlier_idx, rvec, tvec = estimate_pose(pts_3d, pts_2d)
    camera_center = -R.T @ t
    print(f"  Camera center C = {camera_center}  (cm)")

    # 2. Reprojection error gate
    mean_err = check_reprojection(pts_3d, pts_2d, inlier_idx, rvec, tvec)

    # 3. Back-project chess base to Z=0
    X_b, Y_b = backproject_base(R, t, BASE_PX[0], BASE_PX[1], camera_center)

    # 4. Height
    height = compute_height(R, t, X_b, Y_b, TOP_PX[0], TOP_PX[1])

    # 5. Distance
    distance = compute_distance(camera_center, X_b, Y_b)

    # 6. Annotated output
    visualize(img, pts_2d, inlier_idx, rvec, tvec,
              BASE_PX, TOP_PX, distance, height, mean_err, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print(f"  DISTANCE : {distance:.2f} cm")
    print(f"  HEIGHT   : {height:.2f} cm")
    print("=" * 60)


if __name__ == "__main__":
    if "--pick" in sys.argv:
        pick_points(IMAGE_PATH)
    else:
        main()
