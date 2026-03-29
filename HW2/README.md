# HW2: Create the Front View of a Circular Painting by Homography

**Course:** Computer Vision and Applications (CI5336701), 2026 Spring, NTUST
**Student ID:** M11415015
**Name:** 張祐誠

---

## Environment

| Item | Version |
|------|---------|
| Python | 3.11.9 |
| OpenCV (`cv2`) | [填入版本，例如 4.11.0] |
| NumPy | [填入版本] |
| OS | Windows |

Install dependencies:
```bash
pip install opencv-python numpy
```

---

## How to Run

```bash
python hw2_template_filled.py
```

Input files (must be in the same folder):
- `1.jpg` — photo of the painting from the right angle
- `2.jpg` — photo of the painting from the left/front angle

Output files generated:
- `M11415015.jpg` — final result: front view, occlusion-free, cropped to circle
- `merged.jpg` — merged image before final circle crop
- `matches.jpg` — corresponding points marked on both images (side-by-side)

---

## Method Description

### Step 1 — Select Matching Points (img1 → img2)

8 corresponding points were identified on both images by detecting the blue
circular frame in each image via HSV colour segmentation, fitting an ellipse,
and sampling 8 evenly-spaced perimeter points starting from the **topmost point**
going clockwise (45° apart).

> The frame appears as an **ellipse** in each photo because a 3D circle viewed
> from an oblique angle projects onto the 2D image plane as an ellipse
> (perspective foreshortening).

| Point | Clock position | img1 (x, y) | img2 (x, y) |
|-------|---------------|-------------|-------------|
| 1 | 12:00 — Top        | (1091, 168) | (1031,  71) |
| 2 | 1:30  — Top-Right  | (1349, 284) | (1359, 203) |
| 3 | 3:00  — Right      | (1465, 565) | (1506, 523) |
| 4 | 4:30  — Bot-Right  | (1370, 845) | (1385, 842) |
| 5 | 6:00  — Bottom     | (1121, 961) | (1068, 974) |
| 6 | 7:30  — Bot-Left   | ( 862, 845) | ( 740, 842) |
| 7 | 9:00  — Left       | ( 746, 565) | ( 593, 523) |
| 8 | 10:30 — Top-Left   | ( 841, 284) | ( 714, 203) |

### Step 2 — Compute Homography (img1 → img2)

A single 3×3 homography matrix **H** maps img1's coordinate frame to img2's.
It encodes the full perspective distortion between the two viewpoints: scale,
rotation, shear, and perspective foreshortening — all in one matrix.

`cv2.findHomography()` with RANSAC (threshold 5 px) solves for H such that:

```
x_img2 = H · x_img1   (homogeneous coordinates)
```

### Step 3 — Warp img1 into img2's Frame (`warpPerspective`)

`cv2.warpPerspective(img1, H, output_size)` fills every output pixel by
**inverse-mapping**: for each output position `(x', y')`, it computes the source
position using `H⁻¹` and samples `img1` there. This avoids holes that would
appear with forward mapping.

The homogeneous multiply and perspective divide:

```
[x', y', w']ᵀ = H · [x, y, 1]ᵀ
actual coords = (x'/w', y'/w')
```

The division by `w'` (perspective divide) is the step affine transforms cannot
perform. After warping, `warped1` and `img2` share the same coordinate frame and
their circular frames are aligned.

### Step 4 — Merge to Remove Occluding People

`img2` is used as the base (frontal view is clean except for the bald head at
the bottom). The head region is replaced with pixels from `warped1`:

1. **Skin-colour detection** in `img2` using HSV (`H=[7,22], S=[30,220], V=[100,235]`)
2. **ROI restriction** to the lower-centre region (`y > cy+50`, `|x−cx| < 300`)
   to avoid false positives from the cat's orange fur
3. **Convex hull** over all detected skin contours to fill internal gaps
4. **Feathered alpha blend**: the hull mask is Gaussian-blurred (`kernel 61×61`)
   for a smooth edge transition:

```
merged = α × warped1 + (1−α) × img2
```

### Step 5 — Crop to Perfect Circle

A circular mask (centre `(1049, 523)`, radius `454` px — derived from the
detected ellipse in img2) is applied via `cv2.bitwise_and`, then the result is
cropped to the tight bounding box.

---

## Template vs Filled Comparison

The two files differ in exactly three places:

| Location | `hw2_template.py` | `hw2_template_filled.py` |
|----------|-------------------|--------------------------|
| `pts1` / `pts2` | `[0, 0]` placeholders | 8 actual pixel coordinates per image |
| `merge_images` | Simple black-pixel complement + average | Skin detection → convex hull → feathered blend |
| `cx, cy, radius` | `out_w//2`, `out_h//2`, `min//3` (TODOs) | `1049`, `523`, `454` (from ellipse fit) |

`compute_homography`, `warp_to_front`, and `crop_to_circle` required no changes —
their logic was already complete in the template.

---

## Matching Points

![matches](matches.jpg)

---

## Results

| Input 1 | Input 2 |
|---------|---------|
| ![](1.jpg) | ![](2.jpg) |

| Merged | Final Result |
|--------|-------------|
| ![](merged.jpg) | ![](M11415015.jpg) |

---

## Notes / Issues

- Skin-colour ROI thresholds (`cy+50`, `cx±300`) are hand-tuned for these two
  specific images and may not generalise to other inputs.
- The homography maps img1 → img2. The inverse `H⁻¹` maps img2 → img1 if needed.
