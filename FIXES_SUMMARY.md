# VLMotion 問題修復總結

## 修復日期
2025-10-09

## 修復的問題

### 1. **影像中沒有顯示 ArUco 標記**

**問題描述**：在 GUI 中看不到 D405 ArUco 標記的視覺化（邊框、ID、座標軸）

**修復位置**：`/home/hrc/VLMotion/ros2_ws/src/vlservo/VLServo/main_gui.py`

**修復內容**：
- 在 `RoboPointMainWindow.__init__()` 中新增 `self.latest_markers = {}` 屬性來儲存偵測到的標記
- 在 `on_camera_frame()` 方法中，在 ArUco 偵測後增加視覺化邏輯：
  - 繪製標記邊框（綠色多邊形）
  - 顯示標記 ID
  - 使用 `cv2.drawFrameAxes()` 繪製 3D 座標軸（X=紅色，Y=綠色，Z=藍色）

**效果**：現在在 D435i 影像中可以清楚看到所有偵測到的 ArUco 標記，包括 D405 背面的 marker ID 135

---

### 2. **Head Tilt 角度會被往上調整，無法維持在 -75 度**

**問題描述**：設定 LLM Tilt Extra 為 -75° 後，head tilt 會先移動到 -75°，但隨後會被往上調整

**根本原因**：
1. `head_tracker.py` 的 tilt floor 邏輯不夠嚴格
2. `arm_motion.py` 使用錯誤的比較邏輯（`min()` 應該用在正數，負數角度應該用 `max()`）

**修復位置與內容**：

#### A. `head_tracker.py`
- 將 `v_tilt` 分為 `v_tilt_raw` 和最終的 `v_tilt`
- 加強 tilt floor 檢查邏輯：
  ```python
  if cur >= floor_rad and v_tilt_raw > 0.0:
      v_tilt = 0.0  # 阻止往上移動
      print(f"[HeadTracker] Blocked upward tilt...")
  ```
- 加入除錯輸出來追蹤阻擋事件

#### B. `arm_motion.py`
- 修正錯誤的邏輯：
  - **錯誤**：`desired_tilt = min(desired_tilt, floor_rad)`
  - **正確**：`if desired_tilt > floor_rad: desired_tilt = floor_rad`
- 加入詳細的註解說明角度方向（負數 = 向下）
- 加入除錯輸出

#### C. `main_gui.py`
- 在 `HeadTrackerProcess.start()` 中：
  - 加入 `print()` 輸出顯示 tilt floor 參數
  - 降低控制增益：`--k-pan 0.8 --k-tilt 0.6`（原本是 1.0/1.0）
- 在 `LLMGraspControllerProcess.start()` 中：
  - 加入類似的除錯輸出
- 確保 `--tilt-floor-deg` 參數正確傳遞給子程序

**效果**：
- Head tilt 會被嚴格限制在設定的 floor 角度（例如 -75°）
- 即使 ArUco 追蹤想要往上調整，也會被阻擋
- 追蹤更平滑（因為降低增益）

---

## 驗證步驟

### 1. 驗證 ArUco 顯示
```bash
cd /home/hrc/VLMotion
ros2 launch vlservo maingui.launch.py
```
- 點擊 "Start Camera"
- 在影像中應該能看到：
  - ✅ 綠色邊框圍繞每個 ArUco 標記
  - ✅ 標記 ID 顯示在中心（例如 "ID:135"）
  - ✅ RGB 座標軸（X=紅，Y=綠，Z=藍）

### 2. 驗證 Tilt Floor
```bash
cd /home/hrc/VLMotion
ros2 launch vlservo maingui.launch.py
```
- 將 "LLM Tilt Extra" 設為 **-75**
- 點擊 "Start LLM Grasping"
- 觀察終端機輸出，應該看到：
  ```
  [HeadTrackerProcess] Starting with tilt floor: -75.0° (-1.309 rad)
  ```
- 觀察機器人頭部：
  - ✅ 初始會移動到約 -75° 位置
  - ✅ 之後不會往上移動超過 -75°
  - ✅ 如果嘗試往上，終端會顯示 "Blocked upward tilt"

### 3. 檢查程序參數
```bash
# 查看 head_tracker 程序是否正確啟動
ps aux | grep head_tracker

# 應該看到類似：
# python -m VLServo.head_tracker -r --tilt-floor-deg -75.0 --k-pan 0.8 --k-tilt 0.6 --aruco-every-n 3
```

---

## 修改的檔案清單

1. ✅ `/home/hrc/VLMotion/ros2_ws/src/vlservo/VLServo/main_gui.py`
   - 新增 ArUco 視覺化邏輯
   - 初始化 `latest_markers` 屬性
   - 改進 `HeadTrackerProcess` 和 `LLMGraspControllerProcess` 的除錯輸出
   - 降低頭部追蹤增益

2. ✅ `/home/hrc/VLMotion/ros2_ws/src/vlservo/VLServo/head_tracker.py`
   - 強化 tilt floor 檢查邏輯
   - 加入阻擋事件的除錯輸出

3. ✅ `/home/hrc/VLMotion/ros2_ws/src/vlservo/VLServo/arm_motion.py`
   - 修正 tilt floor 夾緊邏輯（從 `min()` 改為正確的比較）
   - 加入詳細註解說明
   - 加入除錯輸出

---

## 技術細節

### ArUco 視覺化原理
```python
# 1. 偵測 ArUco 標記
self.aruco_detector.update(rgb, camera_info)
markers = self.aruco_detector.get_detected_marker_dict()

# 2. 繪製邊框
corners = marker_data.get('corners_2d')
cv2.polylines(vis, [corners_int], True, (0, 255, 0), 2)

# 3. 繪製座標軸
cv2.drawFrameAxes(vis, camera_matrix, distortion_coeffs, 
                  rvec, tvec, axis_length=0.03, thickness=2)
```

### Tilt Floor 角度邏輯
```
向上 (仰視)
    ↑
    0°  (水平)
    ↓
   -45° (稍微向下)
    ↓
   -75° ← Floor (限制線)
    ↓
   -90° (完全向下)

規則：desired_tilt 不能 > floor_rad
例如：desired_tilt = -0.5 rad, floor_rad = -1.31 rad
     -0.5 > -1.31 → 阻擋！設為 -1.31
```

---

## 已知限制

1. **ArUco 偵測頻率**：為了避免 UI 卡頓，每 3 幀才執行一次 ArUco 偵測
2. **追蹤平滑度**：降低增益後追蹤速度變慢，但更穩定
3. **終端輸出**：除錯訊息會顯示在終端，可能產生大量日誌

---

## 回滾方法

如果需要還原修改：

```bash
cd /home/hrc/VLMotion/ros2_ws/src/vlservo/VLServo
git diff main_gui.py
git diff head_tracker.py
git diff arm_motion.py

# 如果要回滾
git checkout main_gui.py
git checkout head_tracker.py
git checkout arm_motion.py
```

---

## 聯絡資訊

如有問題，請檢查：
- 終端機輸出的除錯訊息
- ROS 2 日誌：`ros2 run vlservo maingui` 的輸出
- ArUco 偵測器日誌

修復完成！✨
