# Midterm Project: 3D Model Colorization from Multi-View Images

**Student ID:** M11415015  
**Course:** Computer Vision Application  
**Due:** 2026-04-24

---

## Overview

This project colorizes a 3D point cloud (`Santa.xyz`, 95,406 vertices) by projecting it onto 7 calibration images taken from different viewpoints. The core pipeline is:

1. **Convert point cloud to mesh** (`make_geometry_ply.py`) so MeshLab can pick surface points
2. **Manually pick 2D/3D correspondences** using MeshLab and a custom point-picker
3. **Estimate 3×4 projection matrices** (one per image) via DLT / solvePnPRansac
4. **Colorize vertices** using backface culling, visibility testing, and weighted blending
5. **Write output** as `Santa_colored.ply` (colored mesh for MeshLab)

---

## Method

### Step 0 — Point Cloud to Mesh (`make_geometry_ply.py`)

`Santa.xyz` is a raw point cloud — it has vertices but no surface faces. MeshLab's *Pick Points* tool requires a triangle mesh to snap onto, so the first step is to generate one:

```bash
python make_geometry_ply.py
# output: Santa_geometry.ply
```

`Santa_geometry.ply` is used **only** for picking 3D control points in MeshLab. The actual colorization still operates on the original point cloud `Santa.xyz`.

### Step 1 — 3D Control Point Selection (MeshLab)

Open `Santa_geometry.ply` in MeshLab and use *Filters → Point Set → Pick Points* to select anatomical landmarks. Twelve were chosen:

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

The Direct Linear Transform (DLT) builds a $2N \times 12$ homogeneous system from each correspondence. Each 3D–2D pair $(X, Y, Z) \leftrightarrow (u, v)$ contributes two rows:

$$\begin{bmatrix}
X & Y & Z & 1 & 0 & 0 & 0 & 0 & -uX & -uY & -uZ & -u \\
0 & 0 & 0 & 0 & X & Y & Z & 1 & -vX & -vY & -vZ & -v
\end{bmatrix} \mathbf{p} = \mathbf{0}$$

where $\mathbf{p} \in \mathbb{R}^{12}$ is the row-major flattening of $\mathbf{P}$. Stacking $N$ pairs gives $\mathbf{A}\mathbf{p} = \mathbf{0}$, solved by taking the right singular vector of $\mathbf{A}$ corresponding to the smallest singular value (SVD). Both 3D and 2D points are normalized before solving (Hartley normalization) to improve numerical stability.

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

### Step 3.5 — 3D → 2D Projection (Theory behind `project_points`)

#### Why homogeneous coordinates?

A 3D point $(X, Y, Z)$ is a 3-vector, but $\mathbf{P}$ is $3 \times 4$ — they cannot be multiplied directly.
The fix is to append a $1$, lifting the point into **homogeneous coordinates**:

$$\tilde{\mathbf{X}} = \begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix} \in \mathbb{R}^4$$

This extra $1$ lets translation be absorbed into the matrix multiply. Without it, rotation and translation would have to be applied separately.

#### What does the matrix multiply actually do?

$$\mathbf{proj} = \mathbf{P} \cdot \tilde{\mathbf{X}} = \begin{bmatrix}
s \cdot u \\
s \cdot v \\
s
\end{bmatrix}$$

$\mathbf{P}$ ($3 \times 4$) times $\tilde{\mathbf{X}}$ ($4 \times 1$) gives a 3-vector — but this is **not** a pixel coordinate yet. All three components are scaled by the same depth factor $s$ (the camera-space z value).

#### Perspective division — recovering the actual pixel

$$p_z = \text{proj}[2], \quad u = \frac{\text{proj}[0]}{p_z}, \quad v = \frac{\text{proj}[1]}{p_z}$$

**Why divide by $p_z$?**

This is the core of perspective projection: objects farther away appear smaller. Dividing by depth models that effect. Think of a flashlight: the closer the beam to the wall, the smaller the spot. A camera works the same way — larger $p_z$ shrinks the projected coordinates, making the object occupy fewer pixels.

#### What is inside P?

$$\mathbf{P} = \mathbf{K} \cdot [\mathbf{R} \mid \mathbf{t}]$$

$$\mathbf{K} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}, \quad
[\mathbf{R} \mid \mathbf{t}] = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_1 \\
r_{21} & r_{22} & r_{23} & t_2 \\
r_{31} & r_{32} & r_{33} & t_3
\end{bmatrix}$$

| Component | Role | Shape |
|-----------|------|-------|
| $\mathbf{K}$ | Intrinsics — focal lengths $f_x, f_y$ and principal point $c_x, c_y$ | $3 \times 3$ |
| $\mathbf{R}$ | Camera orientation (rotation) | $3 \times 3$ |
| $\mathbf{t}$ | Camera position (translation) | $3 \times 1$ |
| $\mathbf{P}$ | All of the above fused into one matrix | $3 \times 4$ |

The full chain is: **world coordinates → camera coordinates → image pixel**, done in one shot by $\mathbf{P}$.

#### Vectorized implementation (corresponds to `colorize.py` lines 167, 180–183)

```python
# Process all N vertices at once — no Python for-loop needed
pts_h = np.hstack([pts, np.ones((N, 1))])   # (N, 4)  append 1 to each row
proj  = (P @ pts_h.T).T                      # (N, 3)  matrix multiply
pz    = proj[:, 2]                           # (N,)    depth per vertex
u     = proj[:, 0] / pz                      # (N,)    pixel x
v     = proj[:, 1] / pz                      # (N,)    pixel y
```

**Why two transposes?**
- `pts_h.T` reshapes $(N \times 4)$ to $(4 \times N)$ so $\mathbf{P}$ ($3 \times 4$) can right-multiply it, yielding $(3 \times N)$
- The final `.T` brings it back to $(N \times 3)$ for per-vertex indexing

#### Why the $p_z > 0$ guard?

```python
mask = pz > 0   # keep only vertices in front of the camera
```

If $p_z \leq 0$, the vertex is behind (or exactly on) the camera plane. Dividing by a negative depth flips the projected coordinates, producing completely wrong pixel locations — those vertices must be discarded.

---

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

## 實作問答紀錄

### Q1：在進行 3D 到 2D 的向量化投影時，為什麼需要做兩次轉置 (`.T`)？

**解答：**

這是為了符合矩陣乘法的維度規則，並利用 NumPy 的批次運算加速。

- 投影矩陣 **P**：形狀為 3 × 4
- 齊次座標頂點 **X**：形狀為 N × 4（每列為一個頂點，已附加 1）

為了讓矩陣順利相乘，必須調整維度使其內部對齊 $(3 \times \underline{4}) \times (\underline{4} \times N)$：

1. **第一次轉置**：將頂點矩陣轉為 $\mathbf{X}^\top \in \mathbb{R}^{4 \times N}$
2. **矩陣相乘**：計算 $\mathbf{P} \mathbf{X}^\top$，結果維度為 $(3 \times N)$
3. **第二次轉置**：將結果轉回 $(N \times 3)$，以便直接讀取每一列的 $(s \cdot u, s \cdot v, s)$

$$\mathbf{proj} = (\mathbf{P} \mathbf{X}^\top)^\top = \begin{bmatrix}
s_1 u_1 & s_1 v_1 & s_1 \\
\vdots & \vdots & \vdots \\
s_N u_N & s_N v_N & s_N
\end{bmatrix}$$

對應程式碼（`colorize.py` lines 167, 180–183）：

```python
pts_h = np.hstack([pts, np.ones((N, 1))])   # (N, 4)
proj  = (P @ pts_h.T).T                      # (N, 3)
pz    = proj[:, 2]
u     = proj[:, 0] / pz
v     = proj[:, 1] / pz
```

---

### Q2：背面剔除 (Back-face Culling) 如何透過內積判斷可見性？

**解答：**

定義頂點的單位法向量為 $\mathbf{n}$，從頂點指向相機的視角向量為 $\mathbf{v}$。判定表面「朝向相機」的幾何條件為兩向量夾角 $\theta$ 必須小於 $90^\circ$：

$$\text{Visible} \iff \mathbf{n} \cdot \mathbf{v} > 0 \implies \cos(\theta) > 0$$

若內積 $\leq 0$，則代表該點背對相機（被模型自身遮擋），應停止上色以避免錯誤投影。

```python
view = camera_center_i - vertex_position
if dot(n, view) <= 0:
    skip   # surface faces away from camera
```

---

### Q3：為什麼不直接「四捨五入」像素座標，而要使用雙線性插值？

**解答：**

為了消除**走樣 (Aliasing)** 造成的鋸齒感。如果只取最接近的整數像素，投影後的顏色會出現明顯的馬賽克硬邊。

**雙線性插值 (Bilinear Interpolation)** 會根據小數點距離混合周圍 4 個像素的顏色，讓色彩過渡在 3D 模型表面更為平滑：

| 方法 | 效果 |
|------|------|
| 最近鄰（四捨五入） | 硬邊、馬賽克感、鋸齒明顯 |
| 雙線性插值 | 平滑過渡、色彩連續、視覺品質佳 |

---

### Q4：多視角融合時，為什麼要用「角度」與「距離」計算權重？

**解答：**

為了消除不同照片間的**接縫 (Seams)**。我們根據拍攝品質給予不同的「發言權」：

| 權重因子 | 物理意義 | 目的 |
|---------|---------|------|
| 角度 ($\cos \theta$) | 相機越「正對」表面，投影變形越小 | 優先取正面拍攝的清晰細節 |
| 距離 ($d$) | 離相機越近，取樣解析度通常越高 | 降低遠距離模糊影像的影響 |

融合公式（Lambert cosine weighting）：

```python
w = dot(normalize(n), normalize(view))   # cosine weight
blended = weighted_average(colors, weights)
```

---

## Dependencies

- Python 3.10
- numpy
- Pillow (PIL)
- matplotlib
- opencv-python (`cv2`)
- scipy
