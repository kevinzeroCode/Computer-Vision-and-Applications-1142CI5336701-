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
OUTPUT_PATH = "M11415015.jpg"

# ─── STEP 1: MANUALLY SELECTED MATCHING POINTS ─────────────────────────────────
# 8 points sampled from each detected ellipse, starting from the topmost point,
# going clockwise at 45° intervals (12:00, 1:30, 3:00, 4:30, 6:00, 7:30, 9:00, 10:30)

# Points in image 1 (1.jpg)
pts1 = np.array([
    [1091, 168],  # Top (12:00)
    [1349, 284],  # Top-Right (1:30)
    [1465, 565],  # Right (3:00)
    [1370, 845],  # Bottom-Right (4:30)
    [1121, 961],  # Bottom (6:00)
    [ 862, 845],  # Bottom-Left (7:30)
    [ 746, 565],  # Left (9:00)
    [ 841, 284],  # Top-Left (10:30)
], dtype=np.float32)

# Corresponding points in image 2 (2.jpg) — same physical location on 3D frame
pts2 = np.array([
    [1031,  71],  # Top (12:00)
    [1359, 203],  # Top-Right (1:30)
    [1506, 523],  # Right (3:00)
    [1385, 842],  # Bottom-Right (4:30)
    [1068, 974],  # Bottom (6:00)
    [ 740, 842],  # Bottom-Left (7:30)
    [ 593, 523],  # Left (9:00)
    [ 714, 203],  # Top-Left (10:30)
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
    Merge strategy:
    - warped2 is used as base (frontal view, clean except bald head at bottom)
    - Detect the head in warped2 using skin-colour segmentation (HSV),
      restricted to the lower-centre region to avoid false positives
    - Fill the head region with a convex hull, then blend in warped1 pixels
      via a Gaussian-feathered alpha mask for seamless edges
    """
    canvas_h, canvas_w = warped2.shape[:2]

    # Skin colour range in HSV
    hsv2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv2, np.array([7, 30, 100]), np.array([22, 220, 235]))

    # Restrict to lower-centre region (avoids cat's orange fur)
    cx, cy = canvas_w // 2, canvas_h // 2
    roi = np.zeros((canvas_h, canvas_w), np.uint8)
    roi[cy + 50:, cx - 300:cx + 300] = 255
    skin = cv2.bitwise_and(skin, roi)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    # Convex hull to fill internal gaps
    head_mask = np.zeros((canvas_h, canvas_w), np.uint8)
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillConvexPoly(head_mask, hull, 255)

    # Feathered alpha blend
    alpha  = cv2.GaussianBlur(head_mask.astype(np.float32) / 255.0, (61, 61), 0)
    alpha3 = alpha[..., np.newaxis]
    merged = (alpha3 * warped1.astype(np.float32) +
              (1 - alpha3) * warped2.astype(np.float32)).astype(np.uint8)
    return merged


# ─── STEP 5: CROP TO PERFECT CIRCLE ───────────────────────────────────────────
def crop_to_circle(image, center, radius):
    """
    Mask `image` to a circle defined by `center` (cx, cy) and `radius`.
    Pixels outside the circle are set to black.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Crop tight bounding box around the circle
    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, w)
    y2 = min(center[1] + radius, h)
    cropped = result[y1:y2, x1:x2]
    return cropped


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
        print("\n[Interactive Mode] Click 8 matching points on each image.")
        print("  Pick points on Image 1...")
        p1 = pick_points(img1, "Image 1 — click matching points", n_points=8)
        print("  Pick the SAME points on Image 2...")
        p2 = pick_points(img2, "Image 2 — click matching points", n_points=8)
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

    # ── Define output canvas size (same as img2) ──────────────────────────────
    out_w, out_h = img2.shape[1], img2.shape[0]
    output_size = (out_w, out_h)

    # ── Warp img1 into img2's coordinate frame ────────────────────────────────
    print("\n[Step] Warping Image 1 into Image 2 frame...")
    warped1 = warp_to_front(img1, H_1to2, output_size)

    # ── Merge warped img1 with img2 ───────────────────────────────────────────
    print("\n[Step] Merging images to remove occlusion...")
    merged = merge_images(warped1, img2)
    cv2.imwrite("merged.jpg", merged)
    print("  merged.jpg saved.")

    # ── Crop to circle ────────────────────────────────────────────────────────
    print("\n[Step] Cropping to perfect circle...")
    # Circle centre and radius derived from the detected ellipse in img2
    cx = 1049
    cy = 523
    radius = 454

    result = crop_to_circle(merged, (cx, cy), radius)
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n  Final result saved to {OUTPUT_PATH}")

    # Show result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
