# NTUST Computer Vision and Applications — Homework #1
## AI 協作過程紀錄（Prompt Disclosure）

> **學號：** M11415015
> **課程：** Computer Vision and Applications (CI5336701, 2026 Spring)
> **作業：** HW1 — Leaf Tracking with Camera Projection (`x = K[R|t]X`)

---

## 一、作業概述

本作業目標為：

1. 讀取 `Path.xyz`（3D 葉子落點座標）及 `CameraParamters.yaml`（每幀相機內/外參數）
2. 對每一幀（共 250 幀）套用投影公式 **x = K[R|t]X**，將 3D 座標投影成 2D 像素
3. 在對應影像上繪製標記（圓圈 + 十字）
4. 將所有幀匯出為 MP4 影片

---

## 二、AI 協作流程 & 使用的 Prompts

### Step 1 — 請 AI 產生填空練習模板

**Prompt：**
> 根據 Homework#1.pdf 的作業說明，幫我從 `solution.py` 產生一個「填空練習版本」的 template，
> 關鍵程式碼用 `_______________` 取代，並加上中文 TODO 說明，讓我可以自己練習填寫。

**AI 產出：** `sample_template.py`

- 共 16 個 TODO 填空格（`_______________`）
- 涵蓋 5 個學習模組：XYZ 解析、YAML 讀取、投影計算、標記繪製、VideoWriter 流程
- 每個 TODO 附有中文說明與公式提示

---

### Step 2 — 自行填空完成作業

根據課堂內容與 PDF 說明，將 `sample_template.py` 的空格填入，
另存為 `sample_template copy.py`，嘗試完成作業邏輯。

---

### Step 3 — 請 AI 除錯

**Prompt：**
> 我想問 `sample_template copy.py` 哪裡有問題

**AI 診斷出的 Bugs：**

| 行數 | 問題描述 | 錯誤內容 | 修正方式 |
|------|----------|----------|----------|
| 95 | `v` 的深度除法錯誤 | `p_img[1] / p_img[1]` → 永遠為 1.0 | 改成 `p_img[1] / p_img[2]` |
| 112 | 直線起點 x 座標用錯變數 | `(radius, v - radius)` | 改成 `(u, v - radius)` |
| 150後 | `u, v` 為 float 未轉 int | 直接傳給 OpenCV 函式 | 加 `u, v = int(round(u)), int(round(v))` |

---

### Step 4 — 修正並完成最終版本

根據 AI 的除錯建議，將修正後的邏輯整合至 `solution.py`：

- Bug 1：`v = p_img[1] / p_img[2]`（修正深度 z 的分母）
- Bug 2：直線座標改回 `(u, v - radius)` 與 `(u, v + radius)`
- Bug 3：`project()` 回傳 `int(round(u)), int(round(v))`

---

## 三、檔案結構

```
HW1/
├── solution.py              # 最終完整版本（可直接執行）
├── sample_template.py       # AI 產生的填空模板（16 個 TODO）
├── sample_template copy.py  # 自行填空後的版本（含原始 bugs）
├── Path.xyz                 # 250 幀的 3D 葉子座標
├── CameraParamters.yaml     # 每幀相機內/外參數（FRAME_0000 ~ FRAME_0249）
├── Frames/                  # 250 張原始影像（0000.jpg ~ 0249.jpg）
├── M11415015.mp4            # 輸出影片
└── README.md                # 本文件
```

---

## 四、核心演算法說明

### 投影公式 `x = K[R|t]X`

```
World 座標 X  (4×1 齊次)
        ↓  × [R|t]  (3×4 外參)
Camera 座標 p_cam  (3×1)
        ↓  × K  (3×3 內參)
Image  座標 p_img  (3×1，未正規化)
        ↓  ÷ p_img[2]（深度 z）
Pixel  座標 (u, v)
```

### 關鍵程式碼

```python
p_cam = Rt @ X_hom          # [R|t] · X  (3,4)·(4,) → (3,)
p_img = K  @ p_cam          # K · p_cam  (3,3)·(3,) → (3,)
u = int(round(p_img[0] / p_img[2]))
v = int(round(p_img[1] / p_img[2]))
```

---

## 五、執行方式

將 `Frames/` 資料夾放入 `HW1/` 目錄下，再執行：

```bash
pip install opencv-python numpy pyyaml
python solution.py
```

輸出影片：`M11415015.mp4`

---

## 六、AI 工具使用聲明

本作業使用 **Claude Code (claude-sonnet-4-6)** 作為輔助工具：

- **用途 1：** 自動將完整解答轉換成填空練習模板，供自我學習
- **用途 2：** 對自行填寫後的程式碼進行靜態除錯分析
- **核心邏輯：** 投影公式理解、矩陣運算、OpenCV API 使用均為本人自行學習完成

---

## 七、補充：Windows 中文路徑問題

OpenCV 的 `imread` 在 Windows 上不支援非 ASCII 路徑。
解法：使用 `numpy.fromfile` + `cv2.imdecode` 繞過此限制。

```python
def imread_unicode(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

---

*Generated with [Claude Code](https://claude.ai/claude-code)*
