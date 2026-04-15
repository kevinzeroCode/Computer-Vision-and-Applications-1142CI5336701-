# Midterm Project: 3D Model Colorization from Multi-View Images

Student ID: `M11415015`  
Course: `Computer Vision and Applications`

## What This Folder Contains

This folder contains the full working pipeline for colorizing `Santa.xyz` from 7 images.

Main outputs:
- `Santa_colored.ply`
- `Santa_colored.xyz`

Main scripts:
- `make_geometry_ply.py`
- `pick_2d_points.py`
- `estimate_projection.py`
- `colorize.py`

Required data folders:
- `7Images and xyz/`
- `points/`
- `matrices/`
- `Supplementary/`

## Recommended Reading / Execution Order

If you only want to check the final result:
1. Open `Santa_colored.ply` in MeshLab.

If you want to reproduce the pipeline:
1. Run `make_geometry_ply.py` to create `Santa_geometry.ply` for MeshLab point picking.
2. Use MeshLab to inspect / pick 3D landmarks on `Santa_geometry.ply`.
3. Use `pick_2d_points.py` to click the matching 2D points on each image.
4. Run `estimate_projection.py` to estimate `P01.npy` to `P07.npy`.
5. Check the reprojection errors printed in the terminal.
6. Run `colorize.py` to generate `Santa_colored.ply` and `Santa_colored.xyz`.
7. Open the final `.ply` file in MeshLab for inspection.

## Actual Annotation Workflow Used In This Project

The practical workflow was not “click once and trust it immediately”.

Because manual clicking is noisy, some images were annotated multiple times. The best point set was then retained after checking projection quality and reprojection error. In other words:

1. Pick candidate 2D / 3D correspondences.
2. Estimate projection matrices.
3. Inspect reprojection error.
4. Re-pick weak images if needed.
5. Keep the best version of the points files.
6. Perform the final colorization run.

This is why point quality matters more than simply adding many points.

## Scripts

### `make_geometry_ply.py`

Builds `Santa_geometry.ply` so MeshLab can snap point selection onto a triangle mesh.

```bash
python make_geometry_ply.py
```

### `pick_2d_points.py`

Interactive tool for clicking 2D correspondences.

Controls:
- Left click: add point
- Right click or `z`: undo
- `Enter` or `q`: save

Example:

```bash
python pick_2d_points.py --img "7Images and xyz/01.jpg" --out points/img01_2d.txt
```

### `estimate_projection.py`

Estimates one `3 x 4` projection matrix per image from manual correspondences.

Current final reprojection errors:
- `img01`: `12.76 px`
- `img02`: `10.97 px`
- `img03`: `12.95 px`
- `img04`: `9.65 px`
- `img05`: `4.73 px`
- `img06`: `6.66 px`
- `img07`: `9.69 px`

Run:

```bash
python estimate_projection.py
```

Outputs:
- `matrices/P01.npy` ... `matrices/P07.npy`

### `colorize.py`

Projects all visible images onto the point cloud and writes the final colored model.

Final implementation details:
- Back-face culling
- Z-buffer occlusion filtering
- Bilinear image sampling
- Top-2 view selection instead of averaging all visible views
- Confidence-weighted blending for sharper details

Recommended command:

```bash
python colorize.py
```

Important parameters:
- `--zbuf_eps`: z-buffer tolerance
- `--zbuf_scale`: z-buffer downsample factor
- `--weight_gamma`: stronger preference for frontal views
- `--winner_ratio`: when one view is clearly better, use it directly

Outputs:
- `Santa_colored.ply`
- `Santa_colored.xyz`

## Final Result Summary

With the current final configuration:
- Colored vertices: `95,020 / 95,406`
- Coverage: `99.6%`
- Remaining uncolored vertices use gray fallback

Compared with the earlier baseline, the final version improves:
- hat sharpness
- eye clarity
- visibility handling on self-occluded regions

The lower-body region is the most sensitive area. Adding extra shoe / trouser landmarks can help, but only if those points are manually placed accurately. Poor lower-body correspondences can make the projection worse instead of better.

## Submission Notes

This folder is intended to be directly readable by the reviewer:
- `M11415015_report.pdf` is the required two-page English report.
- `README.md` gives the practical run order.
- `Santa_colored.ply` and `Santa_colored.xyz` are the final outputs.

If the reviewer wants to rerun the pipeline, all required folders are already included.
