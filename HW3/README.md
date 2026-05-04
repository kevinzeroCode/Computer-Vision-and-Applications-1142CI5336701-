# HW3: Chess Piece Distance & Height Estimation

**Course:** Computer Vision and Applications (CI5336701, 2026 Spring)  
**Student:** M11415015

Estimates the **Euclidean distance** from the camera to a chess piece, and the **physical height** of the piece, from a single image using perspective projection.

---

## Files

| File | Description |
|------|-------------|
| `hw3_chess_measurement.py` | Main script |
| `ChessonChecker.png` | Input image (1920×1080) |
| `result.jpg` | Annotated output |
| `report.tex` | LaTeX report |

---

## Usage

```bash
# Run measurement
python hw3_chess_measurement.py

# Interactive point picker (re-select reference corners)
python hw3_chess_measurement.py --pick
```

---

## Results

| Quantity | Value |
|----------|-------|
| Camera center **C** (world) | (−2.05, −12.97, 8.00) cm |
| Piece base (world) | (5.37, −2.80, 0) cm |
| **Distance** *d* | **14.91 cm** |
| **Height** *Z_top* | **3.02 cm** |
| Mean reprojection error | 0.09 px |

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                    │
│   ChessonChecker.png  +  K (intrinsics)  +  12 reference pts   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 │ Camera Pose Estimation                                │
│                                                                 │
│  12 world pts (X,Y,0)  ←→  12 image pts (u,v)                  │
│                                                                 │
│  Harris corner detector  →  cornerSubPix (sub-pixel refine)    │
│          │                                                      │
│          ▼                                                      │
│  solvePnPRansac   →  reject outlier clicks                     │
│          │                                                      │
│          ▼                                                      │
│  solvePnP (L-M)   →  minimize Σ reprojection error²            │
│          │                                                      │
│          ▼                                                      │
│  R, t   →   C = −Rᵀt   (camera center in world coords)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 │ Reprojection Error Check                              │
│                                                                 │
│  Reproject inlier 3D pts → compare to original 2D pts          │
│                                                                 │
│  mean < 2 px  →  OK                                            │
│  mean > 2 px  →  WARNING (continue)                            │
│  mean > 5 px  →  ERROR (stop)                                  │
│                                                                 │
│  Result: mean = 0.09 px, max = 0.21 px  ✓                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 │ Back-project Base Pixel → World Z=0                  │
│                                                                 │
│  base pixel (u_b, v_b) = (1167, 743)                           │
│  [lowest gold pixel via HSV mask, shadow excluded]             │
│                                                                 │
│  Method A (default)  — Ground-plane Homography                  │
│    H = K [r₁ | r₂ | t]                                         │
│    [X_b, Y_b, 1]ᵀ ~ H⁻¹ [u_b, v_b, 1]ᵀ                        │
│                                                                 │
│  Method B (fallback if cond(H) > 1e6) — Ray-Plane Intersect    │
│    d_world = Rᵀ K⁻¹ [u,v,1]ᵀ                                   │
│    λ = −C_z / d_z  →  P = C + λ·d                              │
│                                                                 │
│  Result: (X_b, Y_b) = (5.366, −2.803) cm                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  ┌─────────┴──────────┐
                  │                    │
                  ▼                    ▼
┌────────────────────────┐  ┌──────────────────────────────────────┐
│  STEP 4 │ Height       │  │  STEP 5 │ Distance                   │
│                        │  │                                      │
│  top pixel (u_t, v_t)  │  │  d = ‖C − (X_b, Y_b, 0)ᵀ‖          │
│  = (1179, 387)         │  │                                      │
│  [topmost gold pixel]  │  │  Both C and base are in world        │
│                        │  │  coords (cm) → direct Euclidean      │
│  Assume top = (X_b,    │  │                                      │
│    Y_b, Z_top)         │  │  Result: d = 14.91 cm               │
│  (piece stands upright)│  └──────────────────────────────────────┘
│                        │
│  Solve v-projection:   │
│                        │
│  Z_top = fy·b−(vt−cy)c │
│         ─────────────  │
│         (vt−cy)R₂₂     │
│           − fy·R₁₂     │
│                        │
│  Cross-check w/ u-eq   │
│  (v-eq adopted: 18×    │
│   more stable denom)   │
│                        │
│  Result: Z_top = 3.02cm│
└────────────┬───────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 │ Visualization → result.jpg                           │
│                                                                 │
│  Green dots   : reference corners (original)                   │
│  Yellow rings : reprojected corners (error check)              │
│  Cyan dot     : piece base                                      │
│  Magenta dot  : crown tip                                       │
│  White line   : height indicator                                │
│  Colored arrows: world axes (+X red, +Y green, +Z blue)        │
│  Text overlay : distance, height, mean reprojection error      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Method Summary

### Harris Corner + cornerSubPix
Corner detection is based on the structure tensor of image gradients. For each pixel, build:

$$M = \sum_{W} \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}, \quad R = \det(M) - k\cdot\text{tr}(M)^2$$

Large positive R → corner. `cornerSubPix` then refines to sub-pixel accuracy using local gradient orthogonality constraints.

### Camera Pose (PnP)
$$\{R,\mathbf{t}\} = \arg\min_{R,\mathbf{t}} \sum_i \left\| \begin{bmatrix}u_i\\v_i\end{bmatrix} - \pi(K,R,\mathbf{t},\mathbf{P}_{w,i}) \right\|^2, \qquad \mathbf{C} = -R^\top\mathbf{t}$$

### Height Formula (v-equation)
$$Z_{\text{top}} = \frac{f_y\,b - (v_t - c_y)\,c}{(v_t - c_y)\,R_{22} - f_y\,R_{12}}$$

where $b = R_{1,:}(X_b,Y_b,0)^\top + t_1$, $c = R_{2,:}(X_b,Y_b,0)^\top + t_2$.
