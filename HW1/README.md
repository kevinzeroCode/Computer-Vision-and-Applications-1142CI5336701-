# HW1 - Draw a Marker on a Falling Leaf

NTUST Computer Vision and Applications (CI5336701, 2026 Spring)

## Task

Given 250 consecutive frames of a falling leaf captured by a moving camera, project the 3D leaf position onto each frame using the camera intrinsic and extrinsic parameters, draw a marker at the projected point, and export the result as an MP4 video.

## Files

| File | Description |
|------|-------------|
| `solution.py` | Main program |
| `CameraParamters.yaml` | Camera intrinsic & extrinsic parameters (250 frames) |
| `Path.xyz` | 3D coordinates of the falling leaf (250 points) |
| `Frames/` | Input images `0000.jpg` ~ `0249.jpg` (not tracked by git) |

## Requirements

```bash
pip install opencv-python numpy pyyaml
```

## Usage

Place the `Frames/` folder inside `HW1/`, then run:

```bash
python solution.py
```

Output: `M11415015.mp4` in the same directory.

## How It Works

The projection follows the standard pinhole camera model:

```
x = K [R|t] X
```

- `X` — 3D world coordinate of the leaf (homogeneous `[X, Y, Z, 1]`)
- `[R|t]` — 3×4 extrinsic matrix (rotation + translation), read from YAML
- `K` — 3×3 intrinsic matrix (focal length, principal point), read from YAML
- `x` — projected 2D pixel coordinate after dividing by depth `z`

A circle-cross marker is drawn at `(u, v)` on each frame, then all frames are written to MP4 at 30 fps.

## Note

OpenCV `imread` does not support non-ASCII paths on Windows. The program uses `numpy.fromfile` + `cv2.imdecode` as a workaround.
