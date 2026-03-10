"""
NTUST Computer Vision and Applications - Homework #1
Draw a marker on a falling leaf and export as MP4

Projection formula: x = K[R|t]X
"""

import cv2
import numpy as np
import yaml
import os


def imread_unicode(path):
    """Read image from a path that may contain non-ASCII / CJK characters."""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
XYZ_FILE   = os.path.join(BASE_DIR, "Path.xyz")
YAML_FILE  = os.path.join(BASE_DIR, "CameraParamters.yaml")
FRAMES_DIR = os.path.join(BASE_DIR, "Frames")
OUTPUT_MP4 = os.path.join(BASE_DIR, "M11415015.mp4")   

NUM_FRAMES = 250


# ─── 1. Read 3D leaf positions from Path.xyz ─────────────────────────────────
def load_xyz(path):
    """Return list of (X, Y, Z) tuples, one per frame."""
    points = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            X, Y, Z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append(np.array([X, Y, Z, 1.0]))  # homogeneous
    return points


# ─── 2. Read camera parameters from CameraParamters.yaml ─────────────────────
def load_camera_params(path):
    """Return list of (K, Rt) tuples.
       K  : 3×3 intrinsic matrix
       Rt : 3×4 extrinsic matrix [R|t]
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    params = []
    for i in range(NUM_FRAMES):
        key = f"FRAME_{i:04d}"
        frame_data = data[key]

        K  = np.array(frame_data["intrinsic"],  dtype=np.float64)   # 3×3
        Rt = np.array(frame_data["extrinsic"],  dtype=np.float64)   # 3×4
        params.append((K, Rt))
    return params


# ─── 3. Project 3D point → 2D pixel ──────────────────────────────────────────
def project(K, Rt, X_hom):
    """
    X_hom : (4,) homogeneous world coordinate [X, Y, Z, 1]
    Returns (u, v) pixel coordinates (rounded to int).
    """
    # Camera coordinates (3,)
    p_cam = Rt @ X_hom          # 3×4 · 4 = 3
    # Image coordinates (unnormalized) (3,)
    p_img = K @ p_cam           # 3×3 · 3 = 3
    # Normalise by depth
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    return int(round(u)), int(round(v))


# ─── 4. Draw marker (circle + cross) on image ────────────────────────────────
def draw_marker(img, u, v, radius=18, color=(0, 0, 255), thickness=3):
    """Draw a circle with a cross (+) marker."""
    cv2.circle(img, (u, v), radius, color, thickness)
    cv2.line(img, (u - radius, v), (u + radius, v), color, thickness)
    cv2.line(img, (u, v - radius), (u, v + radius), color, thickness)


# ─── 5. Main pipeline ─────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    leaf_positions  = load_xyz(XYZ_FILE)
    camera_params   = load_camera_params(YAML_FILE)

    # Determine video size from first frame
    first_img = imread_unicode(os.path.join(FRAMES_DIR, "0000.jpg"))
    h, w = first_img.shape[:2]

    # VideoWriter: MP4 with H.264, 30 fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, 30, (w, h))

    print(f"Processing {NUM_FRAMES} frames → {OUTPUT_MP4}")
    for i in range(NUM_FRAMES):
        frame_path = os.path.join(FRAMES_DIR, f"{i:04d}.jpg")
        img = imread_unicode(frame_path)
        if img is None:
            print(f"  [WARN] Cannot read {frame_path}, skipping.")
            continue

        K, Rt   = camera_params[i]
        X_hom   = leaf_positions[i]

        u, v = project(K, Rt, X_hom)

        # Clamp to image boundary before drawing
        if 0 <= u < w and 0 <= v < h:
            draw_marker(img, u, v)
        else:
            print(f"  [WARN] Frame {i:04d}: projected point ({u},{v}) out of image bounds.")

        writer.write(img)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NUM_FRAMES} frames done.")

    writer.release()
    print(f"\nDone! Video saved to: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
