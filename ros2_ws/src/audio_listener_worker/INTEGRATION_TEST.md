# 集成測試指南

此指南說明如何測試 GUI 電腦和麥克風電腦之間的語音輸入系統。

## 環境設定

### GUI 電腦 (A)
- 位置：您當前的工作站
- 運行：`white_point_gui.py`
- 依賴：PyQt5、ROS2、white_point_pipeline

### 麥克風電腦 (B)
- 位置：遠端麥克風設備（如 Stretch 機器人上的麥克風電腦）
- 運行：`audio_listener_worker` ROS2 package
- 依賴：PyAudio、Wit.ai SDK、ROS2

## 網路設定

確保兩台電腦連接到同一個 ROS2 domain（預設 `ROS_DOMAIN_ID=0`）：

### 在兩台電腦上
```bash
# 查看目前的 domain
echo $ROS_DOMAIN_ID

# 如需設定，使用：
export ROS_DOMAIN_ID=0
```

## 測試步驟

### 1. 啟動麥克風節點（電腦 B）

```bash
# 進入 ROS workspace
cd ~/vla/VLMotion/ros2_ws
source install/setup.bash

# 設定 Wit token
export WIT_TOKEN_PATH=/path/to/wit_token.txt

# 啟動節點
ros2 launch audio_listener_worker audio_listener.launch.py
```

預期輸出：
```
[INFO] Audio Listener Worker initialized
[INFO] Wit.ai support: True
[INFO] PyAudio support: True
```

### 2. 啟動 GUI（電腦 A）

在另一個終端運行 GUI：

```bash
cd ~/vla/VLMotion/ros2_ws
source install/setup.bash

# 運行 white_point_gui（根據您的配置調整參數）
ros2 run white_point_pipeline white_point_gui
```

### 3. 測試語音輸入

#### 方式 1：手動測試

1. 在 GUI 視窗中點擊 **Voice Input** 按鈕
2. 按鈕應顯示 "Listening..."
3. 說話（例如 "Move forward"）
4. 等待麥克風端處理並識別
5. 結果應自動填入 Input 欄位

#### 方式 2：使用 ROS CLI 驗證

在第三個終端測試話題通訊：

```bash
# 監視 /speech_to_text 話題
ros2 topic echo /speech_to_text

# 在另一個終端發送 start_listen 訊號
ros2 topic pub /start_listen std_msgs/String "data: start"
```

### 4. 檢查日誌

#### GUI 電腦日誌
```bash
# 查看頻道中的所有訊息
ros2 topic echo /start_listen
```

#### 麥克風電腦日誌
```bash
# 正在運行的節點窗口應顯示
[INFO] Received start_listen signal: start
[INFO] Starting audio recording...
[INFO] Using input device: ...
[INFO] Recording completed. Audio size: ...
[INFO] Sending audio to Wit.ai for recognition...
[INFO] Recognized text: ...
[INFO] Published speech recognition result to /speech_to_text
```

## 常見問題

### GUI 按鈕不工作

**症狀**：點擊 Voice Input 後無反應

**原因**：
- ROS 連接失敗
- 麥克風節點未運行

**解決**：
```bash
# 檢查節點是否運行
ros2 node list

# 檢查話題是否存在
ros2 topic list | grep -E 'start_listen|speech_to_text'

# 檢查網路連接
ping [mikro-computer-ip]
```

### 無法識別語音

**症狀**：麥克風節點收到訊號但未返回識別結果

**原因**：
- Wit token 無效或過期
- 音訊品質不佳
- 網路連接問題

**解決**：
```bash
# 測試 Wit token
python3 -c "from wit import Wit; Wit(open('/path/to/wit_token.txt').read())"

# 檢查音訊輸入
arecord -l  # 列出所有音訊輸入設備
```

### `ModuleNotFoundError: No module named 'speech_recognition_msgs'`

**症狀**：啟動 `audio_listener_node` 時直接崩潰，錯誤顯示找不到 `speech_recognition_msgs`

**原因**：目前容器/主機沒有安裝該 ROS2 message package，或安裝後尚未重新 source 環境

**解決**：
```bash
# 1) 安裝 message package（Humble）
sudo apt-get update
sudo apt-get install -y ros-humble-speech-recognition-msgs

# 2) 重新建置本地 package（建議）
cd ~/vla/VLMotion/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select audio_listener_worker

# 3) 重新 source overlay
source install/setup.bash

# 4) 驗證
ros2 pkg list | grep speech_recognition_msgs
ros2 interface list | grep SpeechRecognitionCandidates
```

若你在 Docker 內操作，請確認執行上面指令的容器就是實際跑 `audio_listener_node` 的那一個容器。

### 麥克風電腦找不到音訊設備

**症狀**：節點報告 "No input device found"

**原因**：
- 麥克風未連接
- 設備驅動未安裝

**解決**：
```bash
# 列出所有音訊設備
aplay -L
arecord -L

# 測試麥克風
arecord -f cd -c1 -d 3 test.wav
aplay test.wav
```

## 性能測試

### 端到端延遲測試

添加時間戳以測量延遲：

```bash
# GUI 端：記錄發送 /start_listen 的時間
# 麥克風端：記錄發送 /speech_to_text 的時間
# 計算差異

# 或使用 ROS 2 中的時間戳
ros2 topic echo /start_listen --no-arr
ros2 topic echo /speech_to_text --no-arr
```

### 語音識別準確度測試

建立測試集並評估：
1. 準備 5-10 句測試句子
2. 依次說出並記錄識別結果
3. 比較預期和實際結果
4. 計算準確度

## 高級配置

### 調整錄音時長

#### 方式 1：啟動時參數化

```bash
ros2 launch audio_listener_worker audio_listener.launch.py \
  record_duration:=5.0
```

#### 方式 2：使用配置文件

```bash
ros2 launch audio_listener_worker audio_listener.launch.py \
  config_file:=/path/to/config/audio_listener.yaml
```

### 更改 Wit 語言

Wit.ai 支援多種語言。編輯 `audio_listener_node.py` 中的 `_recognize_speech` 方法：

```python
# 調整 Wit 請求以指定語言
response = self.wit_client.speech(audio_buffer, {
    "Content-Type": "audio/wav",
    "Accept-Language": "zh-TW"  # 繁體中文
})
```

## 數據記錄

### 保存音訊檔案進行調試

在 `audio_listener_node.py` 中添加：

```python
def _record_audio(self, duration=None, ...):
    ...
    # 在返回之前保存音檔
    with open(f"/tmp/audio_{datetime.now().isoformat()}.wav", "wb") as f:
        f.write(audio_bytes)
    return audio_bytes
```

### 監視所有話題

```bash
ros2 topic list
ros2 topic echo /start_listen
ros2 topic echo /speech_to_text
```

## 定期維護

### 檢查 Wit Token 有效期

Wit token 通常不過期，但某些 API 可能會改變。定期檢查：

1. 登入 [Wit.ai](https://wit.ai/)
2. 驗證帳戶和應用程式狀態
3. 更新 token（如果需要）

### 更新依賴

```bash
# 更新 Wit SDK
pip install --upgrade wit

# 更新 ROS 2 包
sudo apt update && sudo apt upgrade
```

## 文件參考

- [Wit.ai 文件](https://wit.ai/docs)
- [ROS 2 文件](https://docs.ros.org/en/humble/)
- [PyAudio 文件](https://pyaudio.readthedocs.io/)
- [speech_recognition_msgs](https://github.com/ros-drivers/audio_common)
