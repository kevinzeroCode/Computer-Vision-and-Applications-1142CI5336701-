"""
HW2 Solution: Create the front view of a circular painting by homography
NTUST Computer Vision and Applications (CI5336701, 2026 Spring)
Student ID: M11415015

Algorithm Overview:
  1. Detect blue circular frame in both photos using HSV color segmentation + contour fitting
  2. Fit an ellipse to each detected frame (perspective projection of a circle -> ellipse)
  3. Sample 12 corresponding points on each ellipse, aligned to the topmost point
     so both images map the same physical top-of-circle to the top of the target canvas
  4. Compute homography H for each image: ellipse -> canonical frontal circle
  5. Warp both images using their respective H matrices
  6. Merge: use warped2 as base (clean frontal except bottom head),
     detect bald head in lower half via skin-color + convex hull,
     replace with warped1 via feathered alpha blend
  7. Mask result to perfect circle and save

Output files:
  M11415015.jpg  - final result
  warped1.jpg / warped2.jpg - frontal warped views
  merged.jpg     - merged before crop
  matches.jpg    - corresponding points image (for submission)
"""

import cv2
import numpy as np

IMG1_PATH     = "1.jpg"
IMG2_PATH     = "2.jpg"
OUTPUT_PATH   = "M11415015.jpg"
TARGET_RADIUS = 450
PAD           = 60


# -- STEP 1: DETECT BLUE CIRCULAR FRAME -----------------------------------------

def detect_circle_contour(image):
    """
    HSV blue mask -> find most circular large contour.
    Score = (min/max bbox side)^2 * sqrt(area)  -- penalises elongated shapes.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 80, 50]), np.array([135, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_score = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < 200:
            continue
        score = (min(w, h) / max(w, h)) ** 2 * np.sqrt(area)
        if score > best_score:
            best_score, best = score, c
    if best is None or len(best) < 5:
        raise ValueError("Blue circle frame not found")
    return best


# -- STEP 2: HOMOGRAPHY FROM ELLIPSE TO CANONICAL CIRCLE ------------------------

def get_topmost_angle(ellipse):
    """
    Parametric theta of the topmost (min-y) point on the ellipse.

    Derivation:
        y(theta) = cy - a*cos(t)*sin(phi) + b*sin(t)*cos(phi)
        dy/dt    = a*sin(t)*sin(phi) + b*cos(t)*cos(phi) = 0
        => t = atan2(-b*cos(phi), a*sin(phi))

    By aligning theta=0 to the topmost point on BOTH ellipses, we ensure that
    sampling point i on ellipse1 and point i on ellipse2 correspond to the
    same physical angle on the 3D circular frame.
    """
    (cx, cy), (MA, ma), angle_deg = ellipse
    a, b = MA / 2, ma / 2
    phi = np.deg2rad(angle_deg)
    t1 = np.arctan2(-b * np.cos(phi), a * np.sin(phi))
    t2 = t1 + np.pi
    y1 = cy - a * np.cos(t1) * np.sin(phi) + b * np.sin(t1) * np.cos(phi)
    y2 = cy - a * np.cos(t2) * np.sin(phi) + b * np.sin(t2) * np.cos(phi)
    return t1 if y1 < y2 else t2


def sample_ellipse_points(ellipse, n=12):
    """
    Sample n evenly-spaced points on the ellipse starting from the topmost point,
    going clockwise in image coordinates (Y-down).

    OpenCV fitEllipse convention:
      axes = (MA, ma): FULL axis lengths (not semi-axes)
      angle: clockwise rotation from horizontal, in degrees

    Clockwise rotation matrix (Y-down image coords):
        R = [[cos(phi),  sin(phi)],
             [-sin(phi), cos(phi)]]
    """
    (cx, cy), (MA, ma), angle_deg = ellipse
    a, b = MA / 2, ma / 2
    phi = np.deg2rad(angle_deg)
    theta_start = get_topmost_angle(ellipse)
    pts = []
    for i in range(n):
        theta = theta_start + 2 * np.pi * i / n
        px, py = a * np.cos(theta), b * np.sin(theta)
        rx =  px * np.cos(phi) + py * np.sin(phi)   # clockwise rotation
        ry = -px * np.sin(phi) + py * np.cos(phi)
        pts.append([cx + rx, cy + ry])
    return np.array(pts, dtype=np.float32)


def compute_H_ellipse_to_circle(ellipse, target_center, target_radius, n=12):
    """
    Compute homography: ellipse -> canonical circle.
      src: n sampled ellipse points (topmost-aligned, clockwise)
      dst: n points on target circle (from top, clockwise)
    Uses RANSAC for robustness.
    """
    src = sample_ellipse_points(ellipse, n)
    tcx, tcy = target_center
    dst = np.array([
        [tcx + target_radius * np.cos(-np.pi / 2 + 2 * np.pi * i / n),
         tcy + target_radius * np.sin(-np.pi / 2 + 2 * np.pi * i / n)]
        for i in range(n)
    ], dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(f"    Inliers: {int(mask.sum()) if mask is not None else n}/{n}")
    return H


# -- STEP 3: MERGE - REMOVE OCCLUDING PEOPLE ------------------------------------

def remove_head_from_warped2(warped1, warped2, cx, cy, r):
    """
    warped2 gives a clean frontal view of the painting everywhere EXCEPT the
    bald head at the bottom. This function:
      1. Detects the skin-coloured blob only in the lower-centre sub-region
         (avoids false positives from the cat's orange fur higher up)
      2. Fills gaps in the blob using a convex hull
      3. Blends in warped1 pixels (which show the painting there) via a
         Gaussian-feathered alpha mask for seamless edges
    """
    canvas_h, canvas_w = warped2.shape[:2]

    # Skin colour: H=[7,22], moderate-high S, medium-high V
    hsv2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv2, np.array([7, 30, 100]), np.array([22, 220, 235]))

    # Restrict to: below centre-y, centre x +/-250 px, inside circle
    roi = np.zeros((canvas_h, canvas_w), np.uint8)
    cv2.circle(roi, (cx, cy), r, 255, -1)
    roi[:cy + 50, :]  = 0    # below centre only
    roi[:, :cx - 250] = 0    # centre x +/-250 px
    roi[:, cx + 250:] = 0
    skin = cv2.bitwise_and(skin, roi)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    # Convex hull over all skin contour points to fill internal gaps
    head_mask = np.zeros((canvas_h, canvas_w), np.uint8)
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillConvexPoly(head_mask, hull, 255)

    circle_mask = np.zeros((canvas_h, canvas_w), np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 255, -1)
    head_mask = cv2.bitwise_and(head_mask, circle_mask)

    # Feathered alpha blend: smooth transition at mask boundary
    alpha  = cv2.GaussianBlur(head_mask.astype(np.float32) / 255.0, (61, 61), 0)
    alpha3 = alpha[..., np.newaxis]
    merged = (alpha3 * warped1.astype(np.float32) +
              (1 - alpha3) * warped2.astype(np.float32)).astype(np.uint8)
    merged[circle_mask == 0] = 0
    return merged


# -- STEP 4: CROP TO PERFECT CIRCLE --------------------------------------------

def crop_to_circle(image, center, radius):
    """Mask to circle and crop tight bounding box."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    cx, cy, r = int(center[0]), int(center[1]), int(radius)
    return result[max(cy - r, 0):min(cy + r, h),
                  max(cx - r, 0):min(cx + r, w)]


# -- SUBMISSION: MATCHING POINTS IMAGE -----------------------------------------

def save_matches_image(img1, img2, ellipse1, ellipse2, n=8, path="matches.jpg"):
    """
    Create a side-by-side image showing n corresponding points on both input images.
    Points are sampled from each ellipse starting from the topmost point; same index
    = same angular position on the physical 3D circle frame.
    """
    pts1 = sample_ellipse_points(ellipse1, n)
    pts2 = sample_ellipse_points(ellipse2, n)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    scale = 0.4
    i1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
    i2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
    sh1, sw1 = i1.shape[:2]
    sh2, sw2 = i2.shape[:2]

    canvas = np.zeros((max(sh1, sh2), sw1 + sw2 + 10, 3), np.uint8)
    canvas[:sh1, :sw1] = i1
    canvas[:sh2, sw1 + 10:] = i2

    palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
               (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0)]

    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        c = palette[i % len(palette)]
        x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
        x2, y2 = int(p2[0] * scale) + sw1 + 10, int(p2[1] * scale)
        cv2.circle(canvas, (x1, y1), 8, c, -1)
        cv2.circle(canvas, (x2, y2), 8, c, -1)
        cv2.line(canvas, (x1, y1), (x2, y2), c, 1)
        cv2.putText(canvas, str(i + 1), (x1 - 16, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    cv2.imwrite(path, canvas)
    print(f"    {path} saved.")


# -- MAIN ----------------------------------------------------------------------

def main():
    print("=== HW2: Homography Painting Rectification  (M11415015) ===\n")

    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)
    assert img1 is not None and img2 is not None, "Cannot load images"
    print(f"img1: {img1.shape[1]}x{img1.shape[0]},  img2: {img2.shape[1]}x{img2.shape[0]}")

    canvas_size   = TARGET_RADIUS * 2 + PAD * 2
    target_cx     = TARGET_RADIUS + PAD
    target_cy     = TARGET_RADIUS + PAD
    target_center = (target_cx, target_cy)

    # 1. Detect blue ellipses
    print("\n[1] Detecting blue circular frames...")
    e1 = cv2.fitEllipse(detect_circle_contour(img1))
    e2 = cv2.fitEllipse(detect_circle_contour(img2))
    print(f"    Ellipse1: center={tuple(int(v) for v in e1[0])}, "
          f"axes=({e1[1][0]:.0f},{e1[1][1]:.0f}), angle={e1[2]:.1f}")
    print(f"    Ellipse2: center={tuple(int(v) for v in e2[0])}, "
          f"axes=({e2[1][0]:.0f},{e2[1][1]:.0f}), angle={e2[2]:.1f}")

    # 2. Homographies: each ellipse -> canonical frontal circle
    print("\n[2] Computing homographies...")
    H1 = compute_H_ellipse_to_circle(e1, target_center, TARGET_RADIUS, n=12)
    H2 = compute_H_ellipse_to_circle(e2, target_center, TARGET_RADIUS, n=12)

    # 3. Warp both images to frontal view
    print("\n[3] Warping images to frontal view...")
    warped1 = cv2.warpPerspective(img1, H1, (canvas_size, canvas_size))
    warped2 = cv2.warpPerspective(img2, H2, (canvas_size, canvas_size))
    cv2.imwrite("warped1.jpg", warped1)
    cv2.imwrite("warped2.jpg", warped2)
    save_matches_image(img1, img2, e1, e2, n=8)
    print("    warped1.jpg, warped2.jpg, matches.jpg saved.")

    # 4. Merge: remove occluding people
    print("\n[4] Merging (removing head occlusion from warped2)...")
    merged = remove_head_from_warped2(
        warped1, warped2, target_cx, target_cy, TARGET_RADIUS)
    cv2.imwrite("merged.jpg", merged)
    print("    merged.jpg saved.")

    # 5. Crop to perfect circle
    print("\n[5] Cropping to perfect circle...")
    result = crop_to_circle(merged, target_center, TARGET_RADIUS)
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n    Saved: {OUTPUT_PATH}  ({result.shape[1]}x{result.shape[0]} px)")

    # Display
    print("\nDisplaying results -- press any key to close.")
    for title, img in [("Warped 1", warped1), ("Warped 2", warped2),
                       ("Merged",   merged),   ("Result",  result)]:
        s = min(800 / img.shape[1], 600 / img.shape[0], 1.0)
        cv2.imshow(title, cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
