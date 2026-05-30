# Final Project — 3D Reconstruction from a Slit-Laser Turntable Scanner

Reconstructs a colored 3D point cloud of the gnome from 180 laser-scan frames
(2° turntable steps) by **laser-plane triangulation**, and evaluates it against
`GroundTruth.ply`.

**Student ID:** M11415015 · Python 3.11 · NumPy, OpenCV, SciPy, Matplotlib (no open3d needed)

## Files
| file | role |
|------|------|
| **`pipeline.py`** | **Single-file program** — runs all three stages in one go |
| `calibrate.py`   | Camera extrinsics + focal (PnP on the turntable hexagon) and laser-plane fit → `calib.npz` |
| `reconstruct.py` | Stripe detection, ray–plane triangulation, de-rotation, cleanup → `M11415015.xyz` / `.ply` |
| `error_map.py`   | **Evaluation only** — compares the cloud to `GroundTruth.ply` (the only place GT is read) |
| `report/report.tex` | 4-page report |

The single-file `pipeline.py` and the three separate scripts produce identical
results — use whichever is more convenient.

## Run
```bash
# Option A — one file, end to end:
python pipeline.py            # calibrate + reconstruct + evaluate
python pipeline.py --no-eval  # skip the GroundTruth comparison

# Option B — the three separate stages:
python calibrate.py     # -> calib.npz (+ calib_overlay.png)
python reconstruct.py   # -> M11415015.xyz, M11415015.ply   (needs calib.npz)
python error_map.py     # -> error_map.ply, error_hist.png + printed stats
```
Input images are read from `ScandatawithLaser/####.jpg` and
`ScandatawithLaser/withoutLaser/####.jpg`.

## Method (summary)
- **World frame:** origin = turntable centre, +Z = rotation axis. Turntable =
  regular hexagonal prism, circum-radius 25 mm, top z=+1, bottom z=−14.
- **Calibration:** segment the prism silhouette in `withoutLaser/0000.jpg`,
  locate 6 corners (line-fit intersections), match to the hexagon, `solvePnP`.
  The provided `f=720` is inconsistent with the imaged 50 mm table; PnP yields
  **f ≈ 1281 px** (reproj RMS 0.04 px), cx=360, cy=640. Camera at ≈116 mm.
- **Laser plane:** contains the z-axis (`a·x + b·y = 0`); fit from the stripe on
  the bare top face → `−0.504x − 0.864y = 0` (residual 0.30 mm).
- **Detection:** excess-green `2G − R − B` of `(withLaser − withoutLaser)`;
  per-row sub-pixel stripe centroid. (The green laser is absorbed by the **red**
  hat, so `G − max(R,B)` misses ~half the hat; `2G − R − B` recovers it.)
- **Triangulation:** back-project each stripe pixel, intersect the laser plane,
  color from `withoutLaser`, de-rotate by `Rz(−2°·i)` (CCW, measured from the
  images), accumulate; voxel down-sample (0.4 mm) + kNN + radius-outlier removal.

## Result
42,341 colored points. **recon→GT median 0.18 mm, mean 0.20 mm, 95% 0.42 mm.**
Coverage 78.3% of GT within 1 mm; uncovered regions are intrinsic single-view
occlusions (deep concavities, under the arms/sign, turntable underside).

Output format `M11415015.xyz`: `x y z r g b` per line.
