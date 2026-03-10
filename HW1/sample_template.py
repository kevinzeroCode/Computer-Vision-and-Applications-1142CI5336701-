"""
NTUST Computer Vision and Applications - Homework #1
Fill-in-the-blank practice template

Learning objectives (from PDF hints):
  1. TEXT parser: read XYZ and YAML files
  2. Systematically process consecutive images and export as video
  3. Understand and implement x = K[R|t]X
  4. Draw a marker on the projected 2D point
"""

import cv2
import numpy as np
import yaml
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
XYZ_FILE   = os.path.join(BASE_DIR, "Path.xyz")
YAML_FILE  = os.path.join(BASE_DIR, "CameraParamters.yaml")
FRAMES_DIR = os.path.join(BASE_DIR, "Frames")
OUTPUT_MP4 = os.path.join(BASE_DIR, "output.mp4")
NUM_FRAMES = 250


# ════════════════════════════════════════════════════════════════════════
# PART 1 ── TEXT Parser：讀取 XYZ 檔案
# 學習重點：如何開檔、逐行解析、字串分割、型別轉換
# ════════════════════════════════════════════════════════════════════════
def load_xyz(path):
    points = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # TODO 1：用 split(",") 把一行拆成三個字串，並轉成 float
            parts = _______________
            X = _______________
            Y = _______________
            Z = _______________

            # TODO 2：把 (X, Y, Z) 組成齊次座標 [X, Y, Z, 1]（用 np.array）
            points.append(_______________)

    return points  # list of np.array shape (4,)


# ════════════════════════════════════════════════════════════════════════
# PART 2 ── TEXT Parser：讀取 YAML 檔案
# 學習重點：yaml.safe_load、字典取值、轉成 NumPy 矩陣
# ════════════════════════════════════════════════════════════════════════
def load_camera_params(path):
    # TODO 3：用 yaml.safe_load(f) 讀入整個 YAML，存到變數 data
    with open(path, "r") as f:
        data = _______________

    params = []
    for i in range(NUM_FRAMES):
        # TODO 4：組出 key 字串，格式為 "FRAME_0000", "FRAME_0001", ...
        key = _______________
        frame_data = data[key]

        # TODO 5：從 frame_data 取出 "intrinsic" 和 "extrinsic"，轉成 np.array (dtype=float64)
        K  = _______________   # shape (3, 3)
        Rt = _______________   # shape (3, 4)

        params.append((K, Rt))
    return params


# ════════════════════════════════════════════════════════════════════════
# PART 3 ── 投影公式 x = K[R|t]X
# 學習重點：矩陣乘法、齊次座標、除以深度 z 得到像素座標
# ════════════════════════════════════════════════════════════════════════
def project(K, Rt, X_hom):
    """
    X_hom : np.array shape (4,) = [X, Y, Z, 1]
    回傳   : (u, v) 像素座標 (int)

    公式拆解：
        p_cam = [R|t] · X_hom        ← 世界座標 → 相機座標
        p_img = K     · p_cam        ← 相機座標 → 影像平面（未正規化）
        u = p_img[0] / p_img[2]      ← 除以深度 z，得到真正的像素 u
        v = p_img[1] / p_img[2]
    """
    # TODO 6：計算相機座標（矩陣乘法用 @ 運算子）
    p_cam = _______________   # (3,4) @ (4,) → (3,)

    # TODO 7：計算影像平面座標（未正規化）
    p_img = _______________   # (3,3) @ (3,) → (3,)

    # TODO 8：除以深度 z，得到像素座標，並四捨五入成 int
    u = _______________
    v = _______________

    return u, v


# ════════════════════════════════════════════════════════════════════════
# PART 4 ── 畫標記（Marker）
# 學習重點：OpenCV 的 circle / line 函式
# ════════════════════════════════════════════════════════════════════════
def draw_marker(img, u, v, radius=18, color=(0, 0, 255), thickness=3):
    # TODO 9：畫一個圓圈
    # cv2.circle(img, center, radius, color, thickness)
    _______________

    # TODO 10：畫十字（兩條線，一橫一直）
    # cv2.line(img, pt1, pt2, color, thickness)
    _______________   # 橫線
    _______________   # 直線


# ════════════════════════════════════════════════════════════════════════
# PART 5 ── 連續影像處理 + 匯出 MP4
# 學習重點：VideoWriter 初始化、迴圈讀圖、逐幀寫入、release
# ════════════════════════════════════════════════════════════════════════
def imread_unicode(path):
    """Windows 中文路徑讀圖的 workaround"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    print("Loading data ...")
    leaf_positions = load_xyz(XYZ_FILE)
    camera_params  = load_camera_params(YAML_FILE)

    first_img = imread_unicode(os.path.join(FRAMES_DIR, "0000.jpg"))
    h, w = first_img.shape[:2]

    # TODO 11：建立 VideoWriter
    # 參數：輸出路徑、fourcc codec、fps、(寬, 高)
    # fourcc 使用 cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = _______________
    writer = _______________

    print(f"Processing {NUM_FRAMES} frames ...")
    for i in range(NUM_FRAMES):
        # TODO 12：組出圖片路徑（格式：0000.jpg, 0001.jpg, ...）並讀入
        frame_path = _______________
        img = imread_unicode(frame_path)

        K, Rt = camera_params[i]
        X_hom = leaf_positions[i]

        # TODO 13：呼叫 project() 取得 (u, v)
        u, v = _______________

        # 只在點在畫面內才畫
        if 0 <= u < w and 0 <= v < h:
            # TODO 14：呼叫 draw_marker()
            _______________

        # TODO 15：把這幀寫入 VideoWriter
        _______________

    # TODO 16：釋放 VideoWriter（不 release 會導致 mp4 損毀）
    _______________
    print(f"Done! → {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
