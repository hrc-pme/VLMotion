# Audio Listener Worker

遠端麥克風節點，用於在另一台電腦上錄音並進行語音識別。

## 功能流程

### (A) GUI 電腦
1. 用戶按下 `voice_button`
2. 發送 `/start_listen` 訊號

### (B) 麥克風電腦
1. 收到 `/start_listen` 訊號
2. 開始錄音（使用 PyAudio）
3. 語音識別（使用 Wit.ai）
4. 發送 `/speech_to_text` 結果

### (A) GUI 電腦
1. 收到 `/speech_to_text` 訊號
2. 自動填入並發送輸入

## 安裝需求

在麥克風電腦上安裝以下依賴：

```bash
# PyAudio（音訊錄製）
sudo apt-get install python3-pyaudio

# 或通過 pip 安裝（需要安裝開發文件）
sudo apt-get install portaudio19-dev
pip install PyAudio

# Wit.ai 客戶端
pip install wit

# ROS2 依賴
sudo apt-get install ros-humble-speech-recognition-msgs
```

## 建置

在 ROS2 workspace 中建置：

```bash
cd ~/vla/VLMotion/ros2_ws
colcon build --packages-select audio_listener_worker
source install/setup.bash
```

## 使用

### 啟動節點

#### 方式 1：使用 launch 文件

```bash
ros2 launch audio_listener_worker audio_listener.launch.py
```

#### 方式 2：直接執行節點

```bash
ros2 run audio_listener_worker audio_listener_node
```

### 參數設定

通過 launch 文件或啟動時可配置以下參數：

- `record_duration`（預設：4.0 秒）- 錄音時長
- `sample_rate`（預設：16000 Hz）- 採樣率
- `chunk_size`（預設：1024）- 音頻塊大小
- `ws_client_enabled`（預設：true）- 是否啟用 rosbridge websocket client
- `ws_url`（預設：`ws://10.0.0.3:9090`）- rosbridge 位址
- `ws_start_listen_topic`（預設：`/start_listen`）- 透過 websocket 訂閱的觸發 topic
- `ws_speech_to_text_topic`（預設：`/speech_to_text`）- 透過 websocket 回送結果 topic
- `ws_reconnect_sec`（預設：2.0）- websocket 斷線重連間隔
- `use_ros_topic_trigger`（預設：false）- 是否同時啟用 ROS `/start_listen` 訂閱
- `use_ws_trigger`（預設：true）- 是否啟用 websocket `/start_listen` 訂閱
- `use_ros_topic_result_publish`（預設：false）- 是否發布結果到 ROS `/speech_to_text`
- `use_ws_result_publish`（預設：true）- 是否只透過 websocket 回傳 `/speech_to_text`

#### 範例：自訂錄音時長

```bash
ros2 run audio_listener_worker audio_listener_node \
  --ros-args -p record_duration:=5.0
```

## ROS 話題

### 訂閱

- **`/start_listen`**（`std_msgs/String`）
  - GUI 端發送的開始錄音訊號
  - 任何 data 都會觸發錄音

### 發布

- **`/speech_to_text`**（`speech_recognition_msgs/SpeechRecognitionCandidates`）
  - 語音識別結果
  - `transcript`：識別的文字列表
  - `confidence`：信心度列表

## WebSocket Client 模式（你這台當 client）

如果 `rosbridge_websocket_launch.xml` 已經在伺服端啟動，本節點可直接作為 client 連到：

`ws://10.0.0.3:9090`

行為如下：

1. 透過 websocket 訂閱 `/start_listen`（`std_msgs/String`）
2. 收到訊號後在本機錄音 + Wit.ai 辨識
3. 透過 websocket 發布 `/speech_to_text`（`speech_recognition_msgs/SpeechRecognitionCandidates`）

啟動方式：

```bash
ros2 launch audio_listener_worker audio_listener.launch.py
```

若要臨時指定 websocket 位址：

```bash
ros2 run audio_listener_worker audio_listener_node \
  --ros-args \
  -p ws_client_enabled:=true \
  -p ws_url:=ws://10.0.0.3:9090
```

## Wit Token 配置

### 方式 1：環境變數

```bash
export WIT_TOKEN_PATH=/path/to/wit_token.txt
ros2 run audio_listener_worker audio_listener_node
```

### 方式 2：預設位置

系統會在 `/workspace/wit_token.txt` 尋找 token

### 方式 3：獲取 Wit Token

1. 到 [Wit.ai](https://wit.ai/) 建立帳號
2. 建立應用程式
3. 在 Settings → API Details 中複製 Server Access Token
4. 將 token 保存到指定的路徑

## 故障排除

### 找不到麥克風

節點會自動搜尋可用的音訊輸入設備。檢查日誌以確認：

```bash
ros2 run audio_listener_worker audio_listener_node
# 尋找 "Using input device" 的日誌訊息
```

### Wit.ai 識別失敗

- 確認 Wit token 正確
- 確認音檔格式為 WAV、16kHz、16-bit PCM
- 檢查網路連接

### PyAudio 相關錯誤

```bash
# 重新安裝 PyAudio
pip uninstall PyAudio
pip install PyAudio --no-binary :all:
```

## 日誌輸出示例

```
[INFO] Audio Listener Worker initialized
[INFO] Wit.ai support: True
[INFO] PyAudio support: True
[INFO] Received start_listen signal: start
[INFO] Starting audio recording...
[INFO] Using input device: Webcam HD (index: 2)
[INFO] Recording: 4.0s @ 16000Hz, device_index=2
[INFO] Recording completed. Audio size: 128000 bytes
[INFO] Sending audio to Wit.ai for recognition...
[INFO] Recognized text: hello world
[INFO] Published speech recognition result to /speech_to_text
```

## 配置文件（可選）

可在 ROS 參數配置文件中設定：

```yaml
# config/audio_listener.yaml
audio_listener_worker:
  ros__parameters:
    record_duration: 5.0
    sample_rate: 16000
    chunk_size: 1024
```

然後使用：

```bash
ros2 launch audio_listener_worker audio_listener.launch.py \
  config_file:=/path/to/config/audio_listener.yaml
```

## 許可證

MIT
