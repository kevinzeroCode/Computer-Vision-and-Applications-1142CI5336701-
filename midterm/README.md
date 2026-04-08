# Midterm Project: 3D Model Colorization from Multi-View Images

**Student ID:** M11415015  
**Course:** Computer Vision Application  
**Due:** 2026-04-24

---

## Overview

This project colorizes a 3D point cloud (`Santa.xyz`, 95,406 vertices) by projecting it onto 7 calibration images taken from different viewpoints. The core pipeline is:

1. **Manually pick 2D/3D correspondences** using MeshLab and a custom point-picker
2. **Estimate 3×4 projection matrices** (one per image) via DLT / solvePnPRansac
3. **Colorize vertices** using backface culling, visibility testing, and weighted blending
4. **Write output** as `Santa_colored.ply` (colored mesh for MeshLab)

---

## Method

### Step 1 — 3D Control Point Selection (MeshLab)

The 3D model was first converted from a point cloud to a mesh (`make_geometry_ply.py`) so MeshLab's *Pick Points* tool could be used. Twelve anatomical landmarks were selected:

| Name | X | Y | Z |
|------|---|---|---|
| hat_tip | 0.36 | -12.71 | 68.03 |
| right_eye | 8.81 | 1.61 | 40.39 |
| left_eye | -8.50 | 1.18 | 39.64 |
| nose | -0.23 | 7.12 | 40.58 |
| right_shoulder | 13.15 | 7.38 | 25.70 |
| left_shoulder | -12.17 | 8.66 | 26.03 |
| welcome_top_R | 10.55 | 14.15 | 20.53 |
| welcome_top_L | -9.94 | 13.72 | 20.63 |
| welcome_bot_R | 10.39 | 15.19 | 11.26 |
| welcome_bot_L | -9.95 | 15.45 | 11.52 |
| foot_R | 9.29 | 8.89 | 1.01 |
| foot_L | -8.77 | 8.87 | 0.09 |

A subset of these landmarks was selected per image based on visibility.

### Step 2 — 2D Point Picking (`pick_2d_points.py`)

An interactive matplotlib-based tool was used to click the corresponding 2D pixel locations in each image. Left-click adds a point; right-click or `z` undoes; Enter/`q` saves.

### Step 3 — Projection Matrix Estimation (`estimate_projection.py`)

Given N ≥ 6 pairs of 2D-3D correspondences, a 3×4 projection matrix **P** is estimated.

**Algorithm:**

The Direct Linear Transform (DLT) builds a 2N×12 homogeneous system from each correspondence:

```
[X Y Z 1  0 0 0 0  -uX -uY -uZ -u] [p]   [0]
[0 0 0 0  X Y Z 1  -vX -vY -vZ -v]     = [0]
```

The solution is the right singular vector of **A** corresponding to the smallest singular value. Both 3D and 2D points are normalized before solving (Hartley normalization) to improve numerical stability.

**Correspondence recovery:** Since the user may click points in a different order than the 3D file, the algorithm uses:
- **Group-aware permutation search**: points are grouped by similar z-height; the correct v-rank is fixed by the z→v constraint (higher 3D height = smaller v in image), and all permutations within each z-group are tried.
- **solvePnPRansac** (OpenCV): for robustness against outlier correspondences, solvePnPRansac is used with assumed camera intrinsics (focal length: 3000–5000 px, principal point: image center). Multiple focal lengths are tried.

**Reprojection errors and estimated camera centers:**

| Image | Camera Center (X, Y, Z) | Mean Error |
|-------|------------------------|------------|
| img01 | (84, -2, 53) | 127 px |
| img02 | (-83, 1, 65) | 120 px |
| img03 | (72, 112, 41) | 60 px |
| img04 | (-91, -91, -10) | 53 px |
| img05 | (-9, 131, 30) | 53 px |
| img06 | (-70, -58, 40) | 28 px |
| img07 | (41, -76, 32) | 18 px |

The camera center is computed as **C = −P[:,:3]⁻¹ P[:,3]**.

### Step 4 — Colorization (`colorize.py`)

For each vertex **v** with position **(X, Y, Z)** and normal **n**:

```
colors, weights = [], []
for each image i:
    p = P_i @ [X, Y, Z, 1]ᵀ
    u, v = p[0]/p[2],  p[1]/p[2]

    # 1. In-bounds check
    if u or v outside image → skip

    # 2. Back-face culling
    view = camera_center_i − vertex_position
    if dot(n, view) ≤ 0 → skip    # surface faces away from camera

    # 3. Bilinear color sampling
    R, G, B = bilinear_sample(image_i, u, v)

    # 4. Weight by viewing angle (Lambert cosine)
    w = dot(normalize(n), normalize(view))
    colors.append((R, G, B));  weights.append(w)

if colors:
    blended = weighted_average(colors, weights)
else:
    blended = (128, 128, 128)   # gray fallback
```

**Key implementation details:**
- Bilinear sampling is fully vectorized using NumPy for efficiency
- Each P matrix is normalized so `‖P[2,:3]‖ = 1` for metric-scale depth
- The z-buffer was omitted for this sparse point cloud (backface culling alone prevents most occlusion errors)

---

## Results

| Metric | Value |
|--------|-------|
| Total vertices | 95,406 |
| Colored vertices | 95,330 (99.9%) |
| Gray fallback | 76 (0.1%) |
| Hat zone mean color | R=128 G=89 B=66 (reddish) |
| Body zone mean color | R=115 G=114 B=100 (neutral) |
| Boot zone mean color | R=67 G=63 B=50 (dark brown) |

The hat region shows the expected reddish hue (red mushroom hat), and the boot region is appropriately dark. Near-complete coverage (99.9%) is achieved across 7 images.

---

## File Structure

```
midterm/
├── 7Images and xyz/
│   ├── Santa.xyz          # 95,406-vertex point cloud (x y z nx ny nz)
│   ├── 01.jpg … 07.jpg    # 7 calibration images (1836×2748)
├── Supplementary/
│   └── SantaTriangle4Test.ply   # Output template with face data
├── points/
│   ├── shared_3d.txt       # 12 shared 3D control points
│   ├── img01_3d.txt … img07_3d.txt   # Per-image 3D subsets
│   └── img01_2d.txt … img07_2d.txt   # Manually clicked 2D coordinates
├── matrices/
│   └── P01.npy … P07.npy   # Estimated 3×4 projection matrices
├── make_geometry_ply.py    # Converts point cloud to mesh for MeshLab
├── pick_2d_points.py       # Interactive 2D point picker
├── estimate_projection.py  # DLT / solvePnP projection matrix estimator
├── colorize.py             # Main colorization pipeline
├── Santa_geometry.ply      # Mesh used for MeshLab point picking
├── Santa_colored.ply       # ← Final deliverable
└── Santa_colored.xyz       # Colored vertex list (x y z R G B A)
```

---

## How to Run

```bash
# 1. Build geometry mesh for MeshLab picking (if needed)
python make_geometry_ply.py

# 2. Pick 2D points interactively (for each image)
python pick_2d_points.py --img "7Images and xyz/01.jpg" --out points/img01_2d.txt

# 3. Estimate projection matrices
python estimate_projection.py

# 4. Colorize
python colorize.py
```

Open `Santa_colored.ply` in MeshLab to inspect the result.

---

## Dependencies

- Python 3.x
- numpy
- Pillow (PIL)
- matplotlib
- opencv-python (`cv2`)
- scipy
