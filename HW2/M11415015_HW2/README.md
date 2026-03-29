# NTUST Computer Vision and Applications — Homework #2
## AI 協作過程紀錄（Prompt Disclosure）

> **學號：** M11415015
> **姓名：** 張祐誠
> **課程：** Computer Vision and Applications (CI5336701, 2026 Spring)
> **作業：** HW2 — Create the Front View of a Circular Painting by Homography

---

## 一、作業概述

本作業目標為：

1. 讀取兩張從不同角度拍攝的圓形畫作照片（`1.jpg`, `2.jpg`）
2. 在兩張圖上各找至少 5 個對應點，計算 Homography 矩陣 H
3. 將 img1 扭曲到 img2 的座標系，合併兩張圖並移除遮擋人物
4. 裁切成完整正圓，輸出 `M11415015.jpg`

---

## 二、執行方式

```bash
pip install opencv-python numpy
python hw2_template_filled.py
```

輸入檔案（需放在同一資料夾）：
- `1.jpg` — 從右側角度拍攝的照片
- `2.jpg` — 從左/正面角度拍攝的照片

輸出檔案：
- `M11415015.jpg` — 最終結果：正面視角、無人物遮擋、裁切成完整圓
- `merged.jpg` — 圓形裁切前的合併圖
- `matches.jpg` — 對應點標注圖（side-by-side）

---

## 三、AI 協作流程 & 使用的 Prompts

### Step 1 — 請 AI 產生填空練習模板

**Prompt：**
> 根據 Homework#2.pdf 的作業說明，幫我生成一個練習模板，關鍵程式碼用 TODO 取代，讓我可以自己練習填寫。

**AI 產出：** `hw2_template.py`

共有 3 個核心 TODO 填空：
1. `pts1` / `pts2` — 兩張圖的對應點座標
2. `merge_images` — 合併策略（移除人物）
3. `cx, cy, radius` — 圓心座標與半徑（用於最終裁切）

---

### Step 2 — 自行填空完成作業

根據課堂內容與 PDF 說明，將 `hw2_template.py` 的空格填入，
另存為 `hw2_template_filled.py`，完成作業邏輯。

---

### Step 3 — 實作過程的問答紀錄

實作過程中遇到的問題與 AI 的解答（詳見下方第五節）。

---

## 四、核心演算法說明

### Step 1 — 選取對應點（img1 → img2）

對兩張圖各自偵測藍色橢圓邊框，利用橢圓參數方程式從頂點順時針取樣 8 個點：

```
x(t) = cx + a·cos(t)·cos(φ) - b·sin(t)·sin(φ)
y(t) = cy + a·cos(t)·sin(φ) + b·sin(t)·cos(φ)
```

| 點號 | 位置 | img1 (x,y) | img2 (x,y) |
|------|------|-----------|-----------|
| 1 | 12:00 Top      | (1091, 168) | (1031,  71) |
| 2 | 1:30  Top-R    | (1349, 284) | (1359, 203) |
| 3 | 3:00  Right    | (1465, 565) | (1506, 523) |
| 4 | 4:30  Bot-R    | (1370, 845) | (1385, 842) |
| 5 | 6:00  Bottom   | (1121, 961) | (1068, 974) |
| 6 | 7:30  Bot-L    | ( 862, 845) | ( 740, 842) |
| 7 | 9:00  Left     | ( 746, 565) | ( 593, 523) |
| 8 | 10:30 Top-L    | ( 841, 284) | ( 714, 203) |

### Step 2 — 計算 Homography（img1 → img2）

```python
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
# x_img2 = H · x_img1
```

H 是 3×3 矩陣，封裝了兩視角間完整的透視畸變（縮放、旋轉、剪切、透視壓縮）。

### Step 3 — Warp（warpPerspective）

```python
warped1 = cv2.warpPerspective(img1, H, (out_w, out_h))
```

對每個輸出像素反查來源（inverse mapping），利用透視除法還原座標：
```
[x', y', w']ᵀ = H · [x, y, 1]ᵀ → 實際座標 = (x'/w', y'/w')
```

### Step 4 — 合併並移除人物

以 img2 為底，偵測並替換頭部區域：

1. **HSV 膚色偵測** `H=[7,22], S=[30,220], V=[100,235]`
2. **ROI 限制** 下半部中間 `y > cy+50, |x−cx| < 300`（避免誤抓貓咪橘毛）
3. **Convex Hull** 填補偵測空洞
4. **Gaussian Feathered Blend**（kernel 61×61）

```python
alpha = cv2.GaussianBlur(head_mask / 255.0, (61, 61), 0)
merged = alpha × warped1 + (1−alpha) × img2
```

### Step 5 — 裁切成正圓

```python
cx, cy, radius = 1049, 523, 454   # 從 img2 橢圓偵測結果取得
crop_to_circle(merged, (cx, cy), radius)
```

---

## 五、實作問答紀錄

### Q1：對應點座標該怎麼找？直覺是在橢圓上取點，但怎麼計算？

**直覺正確**，`cv2.fitEllipse()` 回傳橢圓中心 `(cx, cy)`、半軸 `a, b`、旋轉角 `φ`，
代入參數方程式即可得到任意角度的像素座標，不需手動點選：

```python
px, py = a * np.cos(theta), b * np.sin(theta)
rx =  px * np.cos(phi) + py * np.sin(phi)
ry = -px * np.sin(phi) + py * np.cos(phi)
pts.append([cx + rx, cy + ry])
```

---

### Q2：為什麼作業要求 5 點，但實作用了 8 點？

作業要求「至少 5 點」。選 8 點的理由：

| 考量 | 說明 |
|------|------|
| 覆蓋率 | 每隔 45°，均勻分布整個圓周，不集中在某一側 |
| RANSAC 強健性 | 點越多，越能過濾橢圓偵測誤差造成的 outlier |
| 可驗證性 | 對應時鐘 8 個方位，容易目視確認是否正確 |

5 點也能執行，但若有 1~2 個點偏移，H 就容易跑掉。

---

### Q3：合併策略為什麼不用簡單平均，要用 HSV + Convex Hull + Alpha Blend？

簡單平均會讓兩個人都半透明，反而更難看。改良做法三層設計各有用途：

| 元件 | 用途 | 少了會怎樣 |
|------|------|-----------|
| HSV 膚色偵測 | 精準定位頭部，不影響其他區域 | 不知道要替換哪裡 |
| Convex Hull | 填補頭髮/陰影造成的 mask 破洞 | 替換形狀不完整 |
| Gaussian Feathered Blend | 邊緣平滑過渡，避免硬邊 | 替換區域像貼上去的 |

```
hard mask:   0 0 0 1 1 1      ← 邊界突兀
feathered:   0 0.2 0.5 0.8 1  ← 平滑過渡
```

---

### Q4：最後 TODO 的圓心和半徑怎麼找？

預設值 `out_w//2, out_h//2` 和 `min//3` 是猜測，不準確。
正確做法是從 img2 的橢圓偵測結果取得：

```python
e2 = cv2.fitEllipse(detect_circle_contour(img2))
cx     = int(e2[0][0])                          # = 1049
cy     = int(e2[0][1])                          # = 523
radius = int((e2[1][0] + e2[1][1]) / 4)        # = 454
```

---

## 六、檔案結構

```
M11415015_HW2/
├── hw2_template_filled.py  # 原始碼（完整填空版本）
├── M11415015.jpg           # 最終輸出結果
├── matches.jpg             # 對應點標注圖（side-by-side）
└── README.md               # 本文件
```

---

## 七、成績配分對應

| 配分項目 | 對應實作 | 佔比 |
|---------|---------|------|
| Correct homography estimation | pts1/pts2 + cv2.findHomography | 50% |
| Merge with occlusion-free effect | HSV + Convex Hull + Alpha Blend | 35% |
| Convert to perfect circle | crop_to_circle（橢圓偵測取圓心） | 15% |

---

## 八、AI 工具使用聲明

本作業使用 **Claude Code (claude-sonnet-4-6)** 作為輔助工具：

- **用途 1：** 產生填空練習模板，供自我學習
- **用途 2：** 實作過程中的概念解答（對應點計算、合併策略設計）
- **核心邏輯：** Homography 原理理解、HSV 膚色偵測範圍調參、Alpha Blend 設計均為本人學習後實作

---

*Generated with [Claude Code](https://claude.ai/claude-code)*
