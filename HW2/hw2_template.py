"""
HW2: Create the front view of a circular painting by homography
NTUST Computer Vision and Applications (CI5336701, 2026 Spring)

Task:
  - Two photos of the same circular painting from different angles
  - Find at least 5 matching points manually
  - Compute homography and warp both images to front view
  - Merge to remove occluding people
  - Crop to a perfect circle

Usage:
  1. Run this script to open an interactive point picker
  2. Click matching points on both images
  3. The homography will be computed and the result saved

Output: <Student_ID>.jpg
"""

import cv2
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────────
IMG1_PATH = "1.jpg"  # image where painting is more visible on the right side
IMG2_PATH = "2.jpg"  # image where painting is more visible on the left side
OUTPUT_PATH = "M11415015.jpg"  # TODO: replace with your student ID

# ─── STEP 1: MANUALLY SELECTED MATCHING POINTS ─────────────────────────────────
# Format: each row is (x, y) pixel coordinate in the image
# TODO: Update these with your own selected matching points
#       At least 5 points, more = better homography

# Points in image 1 (1.jpg)
pts1 = np.array([
    # [x, y],   # describe what landmark this is
    [0, 0],     # PLACEHOLDER — replace with real coords
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
], dtype=np.float32)

# Corresponding points in image 2 (2.jpg)
pts2 = np.array([
    # [x, y],   # same landmark as above
    [0, 0],     # PLACEHOLDER — replace with real coords
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
], dtype=np.float32)


# ─── HELPER: Interactive Point Picker ──────────────────────────────────────────
def pick_points(image, window_title, n_points=5):
    """
    Opens a window for manually clicking n_points on the image.
    Returns an array of shape (n_points, 2) with (x, y) coordinates.
    Left-click to add a point. Press 'q' or close window when done.
    """
    points = []
    img_display = image.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img_display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(img_display, str(len(points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_title, img_display)
            print(f"  Point {len(points)}: ({x}, {y})")
            if len(points) >= n_points:
                print(f"  Collected {n_points} points. Press 'q' to continue.")

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 900, 600)
    cv2.setMouseCallback(window_title, mouse_callback)
    cv2.imshow(window_title, img_display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or (len(points) >= n_points and key != 255):
            break
    cv2.destroyWindow(window_title)
    return np.array(points[:n_points], dtype=np.float32)


# ─── STEP 2: COMPUTE HOMOGRAPHY ────────────────────────────────────────────────
def compute_homography(src_pts, dst_pts):
    """
    Compute homography H such that dst ≈ H @ src  (in homogeneous coords).
    Uses RANSAC for robustness if more than 4 point pairs are given.
    """
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum() if mask is not None else len(src_pts)
    print(f"  Homography computed. Inliers: {inliers}/{len(src_pts)}")
    return H


# ─── STEP 3: WARP IMAGE ────────────────────────────────────────────────────────
def warp_to_front(image, H, output_size):
    """
    Warp `image` using homography H into a canvas of `output_size` (w, h).
    Pixels that have no source are 0 (black).
    """
    w, h = output_size
    warped = cv2.warpPerspective(image, H, (w, h))
    return warped


# ─── STEP 4: MERGE TWO WARPED IMAGES (remove occlusion) ───────────────────────
def merge_images(warped1, warped2):
    """
    Simple merge strategy:
      - Where warped1 is black (no data), use warped2, and vice-versa.
      - Where both have data, blend or prefer one.

    TODO: Improve this — e.g. manually mask out the people in each image,
          then combine the unoccluded regions.
    """
    # Convert to float for blending
    w1 = warped1.astype(np.float32)
    w2 = warped2.astype(np.float32)

    # Simple mask: black pixels = missing data
    mask1 = (w1.sum(axis=2) > 10).astype(np.float32)[..., np.newaxis]
    mask2 = (w2.sum(axis=2) > 10).astype(np.float32)[..., np.newaxis]

    # Where only one image has data, use it; where both have data, average
    both = mask1 * mask2
    only1 = mask1 * (1 - mask2)
    only2 = mask2 * (1 - mask1)

    merged = (only1 * w1 + only2 * w2 + both * (w1 + w2) / 2)
    return merged.astype(np.uint8)


# ─── STEP 5: SECOND HOMOGRAPHY — ELLIPSE → PERFECT CIRCLE ────────────────────
TARGET_RADIUS = 450
PAD           = 60

# TODO: Fill in 4 points on the ellipse in img2 (top / right / bottom / left)
ellipse_pts = np.array([
    [0, 0],   # Top    — PLACEHOLDER
    [0, 0],   # Right  — PLACEHOLDER
    [0, 0],   # Bottom — PLACEHOLDER
    [0, 0],   # Left   — PLACEHOLDER
], dtype=np.float32)

# TODO: Fill in the 4 corresponding points on the target circle
#       Hint: if center=(cx, cy) and radius=R, then
#         Top=(cx, cy-R), Right=(cx+R, cy), Bottom=(cx, cy+R), Left=(cx-R, cy)
_r  = TARGET_RADIUS
_cx = TARGET_RADIUS + PAD
_cy = TARGET_RADIUS + PAD
circle_pts = np.array([
    [_cx,      _cy - _r],
    [_cx + _r, _cy     ],
    [_cx,      _cy + _r],
    [_cx - _r, _cy     ],
], dtype=np.float32)


def warp_to_circle(image):
    """
    TODO: Compute H2 using cv2.getPerspectiveTransform(ellipse_pts, circle_pts),
          warp the merged image to the canonical circle canvas,
          apply a circle mask to clean edges, and crop to bounding box.
    """
    canvas_size = TARGET_RADIUS * 2 + PAD * 2
    H2 = cv2.getPerspectiveTransform(ellipse_pts, circle_pts)  # TODO: fill ellipse_pts above
    warped = cv2.warpPerspective(image, H2, (canvas_size, canvas_size))

    mask = np.zeros((canvas_size, canvas_size), np.uint8)
    cv2.circle(mask, (_cx, _cy), TARGET_RADIUS, 255, -1)
    warped = cv2.bitwise_and(warped, warped, mask=mask)

    r = TARGET_RADIUS
    return warped[_cy - r:_cy + r, _cx - r:_cx + r]


# ─── VISUALISE MATCHING POINTS ─────────────────────────────────────────────────
def visualise_matches(img1, img2, p1, p2, save_path="matches.jpg"):
    """
    Draw matching points side-by-side and save for submission.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas = np.zeros((canvas_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255),
              (128, 255, 0), (255, 128, 0)]

    for i, (pt1, pt2) in enumerate(zip(p1, p2)):
        c = colors[i % len(colors)]
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])
        cv2.circle(canvas, (x1, y1), 8, c, -1)
        cv2.circle(canvas, (x2, y2), 8, c, -1)
        cv2.line(canvas, (x1, y1), (x2, y2), c, 1)
        cv2.putText(canvas, str(i + 1), (x1 - 15, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(canvas, str(i + 1), (x2 - 15, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    cv2.imwrite(save_path, canvas)
    print(f"  Matching points saved to {save_path}")
    return canvas


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=== HW2: Homography-based Image Rectification ===\n")

    # Load images
    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)
    assert img1 is not None, f"Cannot read {IMG1_PATH}"
    assert img2 is not None, f"Cannot read {IMG2_PATH}"
    print(f"Image 1 size: {img1.shape[1]}x{img1.shape[0]}")
    print(f"Image 2 size: {img2.shape[1]}x{img2.shape[0]}")

    # ── Option A: Use pre-defined points (fill in pts1, pts2 above) ──────────
    USE_INTERACTIVE = (pts1[0][0] == 0 and pts1[0][1] == 0)  # auto-detect placeholder

    if USE_INTERACTIVE:
        # ── Option B: Interactive point picker ───────────────────────────────
        print("\n[Interactive Mode] Click 5+ matching points on each image.")
        print("  Pick points on Image 1...")
        p1 = pick_points(img1, "Image 1 — click matching points", n_points=5)
        print("  Pick the SAME points on Image 2...")
        p2 = pick_points(img2, "Image 2 — click matching points", n_points=5)
    else:
        p1, p2 = pts1, pts2
        print("Using pre-defined matching points.")

    print(f"\nPoints in img1:\n{p1}")
    print(f"Points in img2:\n{p2}")

    # Save matching visualisation (required for submission)
    print("\n[Step] Visualising matches...")
    visualise_matches(img1, img2, p1, p2, save_path="matches.jpg")

    # ── Compute homography: map img1 → img2 perspective ──────────────────────
    print("\n[Step] Computing homography (img1 → img2)...")
    H_1to2 = compute_homography(p1, p2)
    print(f"  H =\n{H_1to2}")

    # ── Define output canvas size (same as img2 by default) ──────────────────
    out_w, out_h = img2.shape[1], img2.shape[0]
    output_size = (out_w, out_h)

    # ── Warp img1 into img2's coordinate frame ────────────────────────────────
    print("\n[Step] Warping Image 1 into Image 2 frame...")
    warped1 = warp_to_front(img1, H_1to2, output_size)
    cv2.imwrite("warped1.jpg", warped1)
    print("  warped1.jpg saved.")

    # ── Merge warped img1 with img2 ───────────────────────────────────────────
    print("\n[Step] Merging images to remove occlusion...")
    merged = merge_images(warped1, img2)
    cv2.imwrite("merged.jpg", merged)
    print("  merged.jpg saved.")

    # ── H2: warp merged image → perfect circle ───────────────────────────────
    # TODO: fill in ellipse_pts above (4 points on the ellipse in img2)
    print("\n[Step] Applying H2 (ellipse → perfect circle)...")
    result = warp_to_circle(merged)
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n  Final result saved to {OUTPUT_PATH}")

    # Show result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
