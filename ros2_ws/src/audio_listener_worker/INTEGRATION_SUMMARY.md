# 遠端語音輸入系統 - 整合摘要

## 概述

已為您的 VLMotion 系統建立了完整的遠端語音輸入架構。該系統允許在 GUI 電腦上按下語音按鈕，從而在遠端麥克風電腦上觸發錄音和語音識別。

## 建立的文件

### 新 ROS2 Package：`audio_listener_worker`

位置：`/home/hrc/vla/VLMotion/ros2_ws/src/audio_listener_worker/`

**核心文件：**
- `audio_listener_worker/audio_listener_node.py` - 主要節點實現
- `launch/audio_listener.launch.py` - ROS2 launch 文件
- `config/audio_listener.yaml` - 參數配置文件
- `package.xml` - Package 定義
- `setup.py` - Python 安裝配置
- `README.md` - 詳細使用說明
- `INTEGRATION_TEST.md` - 集成測試指南

### 修改的文件

**白點 GUI：**
- `white_point_pipeline/white_point_pipeline/white_point_gui.py`
  - 添加了 `/start_listen` 發行者（第 564 行）
  - 更新了 `handle_voice_button()` 方法以發送訊號（第 1509 行）

## 系統架構

```
┌─────────────────────┐
│   GUI 電腦 (A)      │
├─────────────────────┤
│ white_point_gui.py  │
│  ↓ user press       │
│  → publish          │
│    /start_listen    │
└──────────────┬──────┘
               │ ROS2 network
               ↓
┌──────────────────────────┐
│ 麥克風電腦 (B)           │
├──────────────────────────┤
│ audio_listener_worker    │
│  ↓ receive /start_listen │
│  → record audio          │
│  → recognize speech      │
│  → publish               │
│    /speech_to_text       │
└──────────────┬───────────┘
               │ ROS2 network
               ↓
┌─────────────────────────┐
│   GUI 電腦 (A)          │
├─────────────────────────┤
│ on_ros_speech_text()    │
│  ↓ receive result       │
│  → auto fill input      │
│  → send message         │
└─────────────────────────┘
```

## 工作流程

### 1. 用戶操作（GUI 電腦）
```
點擊 "Voice Input" 按鈕
  ↓
發送 /start_listen 訊號
  ↓
顯示 "Listening..."
  ↓
等待 /speech_to_text 結果
```

### 2. 遠端錄音（麥克風電腦）
```
收到 /start_listen 訊號
  ↓
開始 PyAudio 錄音
  ↓
發送音檔給 Wit.ai 進行識別
  ↓
接收識別結果
  ↓
發布到 /speech_to_text 話題
```

### 3. 結果處理（GUI 電腦）
```
收到 /speech_to_text 訊號
  ↓
自動填入 Input 欄位
  ↓
自動提交給模型
```

## ROS2 話題

### 訂閱和發布

**GUI 電腦（白點 GUI）**
- 發布：`/start_listen` (String)
- 訂閱：`/speech_to_text` (SpeechRecognitionCandidates)

**麥克風電腦（audio_listener_worker）**
- 訂閱：`/start_listen` (String)
- 發布：`/speech_to_text` (SpeechRecognitionCandidates)

## 安裝和配置步驟

### 步驟 1：在麥克風電腦上安裝依賴

```bash
# 系統依賴
sudo apt-get update
sudo apt-get install -y portaudio19-dev

# Python 包
pip install PyAudio wit

# ROS2 依賴（如未安裝）
sudo apt-get install ros-humble-speech-recognition-msgs
```

### 步驟 2：建置 ROS2 Package

```bash
cd ~/vla/VLMotion/ros2_ws
colcon build --packages-select audio_listener_worker
source install/setup.bash
```

### 步驟 3：設定 Wit Token

```bash
# 在麥克風電腦上設定環境變數
export WIT_TOKEN_PATH=/path/to/wit_token.txt

# 或將其添加到 ~/.bashrc
echo 'export WIT_TOKEN_PATH=/path/to/wit_token.txt' >> ~/.bashrc
source ~/.bashrc
```

### 步驟 4：啟動服務

**麥克風電腦：**
```bash
cd ~/vla/VLMotion/ros2_ws
source install/setup.bash
ros2 launch audio_listener_worker audio_listener.launch.py
```

**GUI 電腦：**
```bash
cd ~/vla/VLMotion/ros2_ws
source install/setup.bash
ros2 run white_point_pipeline white_point_gui
```

## 參數配置

### 可調整的參數

編輯 `config/audio_listener.yaml` 或在啟動時指定：

```bash
# 錄音時長（秒）
ros2 launch audio_listener_worker audio_listener.launch.py \
  record_duration:=5.0

# 採樣率（Hz）
ros2 launch audio_listener_worker audio_listener.launch.py \
  sample_rate:=16000

# 音頻塊大小
ros2 launch audio_listener_worker audio_listener.launch.py \
  chunk_size:=1024
```

## 常見命令

### 查看節點狀態

```bash
# 列出所有運行中的節點
ros2 node list

# 查看特定節點的發布/訂閱
ros2 node info /audio_listener_worker
```

### 監視話題

```bash
# 監視 /start_listen 訊號
ros2 topic echo /start_listen

# 監視 /speech_to_text 結果
ros2 topic echo /speech_to_text

# 列出所有話題
ros2 topic list
```

### 手動測試

```bash
# 終端 1：啟動麥克風節點
ros2 launch audio_listener_worker audio_listener.launch.py

# 終端 2：發送開始錄音訊號
ros2 topic pub /start_listen std_msgs/String "data: start"

# 終端 3：監視結果
ros2 topic echo /speech_to_text
```

## 故障排除

### 問題：找不到麥克風設備

**解決方案：**
```bash
# 列出可用的音訊設備
aplay -L
arecord -L

# 編輯 audio_listener_node.py，手動指定設備索引
```

### 問題：Wit.ai 識別失敗

**解決方案：**
1. 驗證 token：`cat $WIT_TOKEN_PATH`
2. 測試網路連接
3. 檢查音檔格式和品質

### 問題：ROS2 通訊超時

**解決方案：**
1. 確認兩台電腦在同一個網路
2. 檢查 `ROS_DOMAIN_ID` 一致性
3. 驗證防火牆設定

## 高級配置

### 使用 Whisper（替代 Wit.ai）

如果要改用 OpenAI Whisper 進行語音識別，可修改 `audio_listener_node.py` 中的 `_recognize_speech` 方法：

```python
import whisper

# 在 __init__ 中加載模型
self.whisper_model = whisper.load_model("base")

# 在 _recognize_speech 中使用
def _recognize_speech(self, audio_bytes):
    # 保存臨時文檔
    with open("/tmp/temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    
    # 使用 Whisper 識別
    result = self.whisper_model.transcribe("/tmp/temp_audio.wav")
    return result["text"]
```

### 自訂音訊設備選擇

編輯 `_record_audio` 方法中的設備選擇邏輯：

```python
# 改變優先設備選擇策略
device_name = device_info.get("name", "").lower()
if "your_device_name" in device_name:
    input_device_index = i
    break
```

## 性能指標

預期延遲：
- 錄音：4 秒（可配置）
- 傳輸和識別：2-5 秒
- 總依端到端：6-9 秒

## 安全性考慮

1. **Wit Token 保護**
   - 不要在代碼中硬編碼 token
   - 使用環境變數或配置文件
   - 定期輪換 token

2. **音訊隱私**
   - 考慮本機儲存識別結果而非原始音檔
   - 實現音訊加密（如需要）

3. **網路安全**
   - 在受信任的網路中使用
   - 考慮使用 VPN 或防火牆保護 ROS2 通訊

## 文件參考

- [完整 README](./README.md)
- [集成測試指南](./INTEGRATION_TEST.md)
- [Wit.ai 文件](https://wit.ai/docs)
- [ROS2 文件](https://docs.ros.org/en/humble/)

## 後續改進建議

1. **增加語言支援** - 添加多語言識別選項
2. **本地識別** - 實現 Whisper 基礎識別以減少網路依賴
3. **識別結果快取** - 避免重複識別相同短語
4. **音訊品質分析** - 在發送前檢查音訊品質
5. **命令別名** - 為常見 VLA 命令創建快捷方式

## 支援和報告問題

如遇問題，請檢查：
1. 節點日誌輸出
2. ROS2 話題連接
3. 依賴軟件版本
4. 網路連接性

---

**系統創建日期：2026-04-08**
**兼容版本：ROS2 Humble+**
