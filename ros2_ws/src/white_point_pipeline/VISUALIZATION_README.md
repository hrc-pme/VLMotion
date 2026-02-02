# White Point Pipeline 視覺化使用說明

## 功能說明

現在系統會在 RViz 中顯示以下視覺化標記：

### 標記說明

1. **🔴 紅色球體**：原始目標點（相機檢測到的白點位置）
2. **🟡 黃色球體**：準備點（approach point，在面板前方 60cm）
3. **🔵 藍色線條**：機器人移動軌跡（base_link 的路徑）
4. **🟢 綠色球體**：當前夾爪位置（link_gripper_fingertip_left）
5. **🟣 洋紅色球體**：補償後的目標點（考慮夾爪 Y 軸偏移後的實際目標）

## 啟動方式

### 方法 1：使用帶視覺化的 Launch 文件（推薦）

```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch white_point_pipeline white_point_with_viz.launch.py
```

這會自動啟動：
- 所有原始節點（stretch_driver, camera, GUI, 控制節點等）
- RViz2 並載入預配置

### 方法 2：分別啟動

終端 1：
```bash
ros2 launch white_point_pipeline white_point_pipeline.launch.py
```

終端 2：
```bash
ros2 run rviz2 rviz2 -d ~/ros2_ws/install/white_point_pipeline/share/white_point_pipeline/config/white_point_visualization.rviz
```

## 使用方法

1. **啟動系統**後，RViz 會自動打開
2. **在 GUI 中點擊白點**
3. **觀察 RViz 中的視覺化**：
   - 紅色球體會出現在目標位置
   - 黃色球體顯示準備點
   - 藍色線條記錄機器人移動路徑
   - 綠色球體跟隨夾爪移動
   - 洋紅色球體顯示考慮偏移後的實際目標

4. **驗證是否到達目標**：
   - 最終綠色球體（夾爪）應該接近紅色球體（原始目標）
   - 查看兩者之間的距離判斷精度

## RViz 檢視技巧

- **旋轉視角**：按住滑鼠中鍵拖曳
- **平移視角**：按住 Shift + 滑鼠中鍵拖曳
- **縮放**：滾動滑鼠滾輪
- **切換顯示**：左側面板可以開關各個標記的顯示

## Topic 資訊

視覺化標記發布到：`/white_point_markers` (MarkerArray)

你也可以用以下命令查看：
```bash
ros2 topic echo /white_point_markers
```

## 調試信息

查看軌跡記錄和標記發布頻率：
```bash
ros2 topic hz /white_point_markers
```

## 顏色含義總結

| 顏色 | 含義 | 說明 |
|------|------|------|
| 🔴 紅色 | 原始目標 | 相機檢測到的白點 |
| 🟡 黃色 | 準備點 | 接近階段的中間點 |
| 🔵 藍色 | 軌跡線 | 機器人底盤移動路徑 |
| 🟢 綠色 | 夾爪位置 | 實時夾爪位置 |
| 🟣 洋紅色 | 補償目標 | 考慮夾爪偏移的實際目標 |

## 故障排除

如果看不到標記：
1. 確認 RViz 中 "Markers" 顯示已啟用
2. 確認 Fixed Frame 設置為 "odom"
3. 檢查 Topic 是否正確：`/white_point_markers`
4. 確認已經點擊了目標點（否則沒有標記）
