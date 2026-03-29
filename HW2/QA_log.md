# HW2 實作問答紀錄

記錄實作過程中遇到的問題與解答。

---

## Q1：座標該怎麼找？直覺是在橢圓上取點，但怎麼計算出座標？

**直覺正確**：確實是在橢圓上取樣點。

`cv2.fitEllipse()` 回傳橢圓參數 `(cx, cy)`、半軸長 `a, b`、旋轉角 `φ`，
代入橢圓參數方程式即可算出任意角度 t 的像素座標：

```
x(t) = cx + a·cos(t)·cos(φ) - b·sin(t)·sin(φ)
y(t) = cy + a·cos(t)·sin(φ) + b·sin(t)·cos(φ)
```

不需要手動點選，全部用數學計算。

對應程式碼（`hw2_template_filled.py`）：

```python
px, py = a * np.cos(theta), b * np.sin(theta)   # 橢圓局部座標
rx =  px * np.cos(phi) + py * np.sin(phi)        # 旋轉回影像座標
ry = -px * np.sin(phi) + py * np.cos(phi)
pts.append([cx + rx, cy + ry])
```

---

## Q2：為什麼從作業要求的 5 點變成 8 點？

作業要求「至少 5 點」，5 是最低門檻（homography 有 8 個自由度，4 對點理論上夠解，
5 點開始有 redundancy 可以用 RANSAC）。

選 8 點的理由：

| 考量 | 說明 |
|------|------|
| 覆蓋率 | 每隔 45°，均勻分布整個橢圓圓周，不會集中在某一側 |
| RANSAC 強健性 | 點越多，越能過濾掉橢圓偵測誤差造成的 outlier |
| 可驗證性 | 對應時鐘 8 個方位（12/1:30/3/4:30/6/7:30/9/10:30），容易目視確認 |

5 點也能執行，但若有 1、2 個點偏移，homography 容易跑掉。
8 點讓 RANSAC 有足夠 inlier 維持準確度。

---

## Q3：合併策略為什麼不用簡單平均，而要用 HSV + Convex Hull + Alpha Blend？

**原始做法的問題：**

兩張圖的人站在不同位置，平均後兩個人都變半透明，反而更難看。

```python
# 簡單平均
merged = only1*w1 + only2*w2 + both*(w1+w2)/2
```

**改良做法的三個優點：**

### ① HSV 膚色偵測 — 精準找到問題區域

不盲目處理整張圖，只針對有問題的頭部區域。其他區域完全保留 img2 原樣。
HSV 的 H channel 把色相獨立出來，膚色的 H 值範圍很窄（7–22），比 RGB 更容易框住。

### ② Convex Hull — 填補偵測空洞

頭髮、眼鏡、陰影會讓膚色 mask 出現破洞。
Convex hull 把所有偵測到的輪廓包成完整凸多邊形，確保整個頭部被覆蓋。

```
膚色 mask（有破洞）  →  convex hull（完整填滿）
  ■ ■   ■              ■ ■ ■ ■ ■
■       ■          →   ■ ■ ■ ■ ■ ■
  ■ ■ ■                ■ ■ ■ ■ ■
```

### ③ Gaussian Feathered Blend — 邊緣無縫

直接 hard mask 替換會有明顯邊界線。Gaussian blur 把 mask 邊緣變成漸層：

```
hard mask:   0 0 0 1 1 1      ← 邊界突兀
feathered:   0 0.2 0.5 0.8 1  ← 平滑過渡

merged = α × warped1 + (1−α) × img2
```

**三者缺一不可：**

| 少了哪個 | 結果 |
|---------|------|
| 沒有 HSV 偵測 | 不知道要替換哪裡，只能盲目平均 |
| 沒有 Convex Hull | 頭部有破洞，替換形狀不完整 |
| 沒有 Feathered Blend | 替換區域邊界有明顯硬邊，像貼上去的 |

---

## Q4：最後一個 TODO 是找圓心和半徑來 crop，要怎麼找？

**TODO 原始佔位：**

```python
cx = out_w // 2          # TODO
cy = out_h // 2          # TODO
radius = min(out_w, out_h) // 3  # TODO
```

**正確做法：從 img2 的橢圓偵測結果取得**

前面已經對 img2 做了橢圓偵測，橢圓的中心和半軸就是答案：

```python
e2 = cv2.fitEllipse(detect_circle_contour(img2))
# e2 = ((cx, cy), (MA, ma), angle)

cx     = int(e2[0][0])
cy     = int(e2[0][1])
radius = int((e2[1][0] + e2[1][1]) / 4)  # 兩軸平均後取半
```

本題 img2 的偵測結果：`center=(1049, 523)`，`semi-axes=(444, 464)`，`radius≈454`

**為什麼不能用預設值？**

| 方法 | 問題 |
|------|------|
| `out_w // 2`, `out_h // 2` | 圓心不一定在圖片正中央 |
| `min(w,h) // 3` | 純粹猜測，幾乎一定裁到邊框或裁太小 |
| 橢圓偵測結果 | 直接從影像計算，準確對應畫框實際位置 |

---
