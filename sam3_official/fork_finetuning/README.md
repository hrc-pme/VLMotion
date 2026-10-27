# SAM3 Fork/Scissors Fine-tuning

本目錄是目前叉子（`fork`）與剪刀（`scissors`）的 SAM3 聯合微調流程。主流程使用 Meta 官方 SAM3 image trainer，資料為 **COCO bbox-only** 格式，訓練完成後再轉成 Hugging Face 格式供 ROS worker 載入。

## 追加叉子托盤資料的聯合續訓

`/workspace/Forks_plate.coco/train` 新增的 5 張影像、45 個叉子標註，全部加入叉子訓練集；剪刀沿用原資料，兩類仍在同一個模型中聯合訓練。原本的測試切分保持不變，新圖不進入測試集。新訓練集包含叉子 33 張、剪刀 22 張；每個 epoch 對每類各取 33 個樣本，因此叉子 33 張每次都會被使用，剪刀則重複取樣以平衡類別。

```bash
bash /workspace/sam3_official/fork_finetuning/run_fork_scissors_joint_plate_v2_full_ft_train.sh
```

此腳本使用 `/workspace/sam3_official/sam3/train/configs/fork/fork_scissors_joint_plate_v2_full_ft.yaml`，從既有的叉子／剪刀聯合微調 checkpoint 續訓 5 epochs，資料在 `/workspace/sam3_official/fork_finetuning/data/ForkScissors.joint.plate_v2.full_ft`，新 checkpoint 儲存於 `/workspace/sam3_official/fork_finetuning/experiments/fork_scissors_joint_plate_v2_full_ft/checkpoints/checkpoint.pt`。依賴套件安裝在 `/workspace/sam3_official/.train_deps`，腳本會設定 `PYTHONPATH`。訓練完成後，須依下方第 4.2 節匯出成 HF 格式，驗證後才替換 worker 權重。

若要從原始 SAM3 權重對相同新版混合資料集完整訓練 20 epochs，使用：

```bash
bash /workspace/sam3_official/fork_finetuning/run_fork_scissors_joint_plate_v4_20ep_full_ft_train.sh
```

此版本的 config 是 `/workspace/sam3_official/sam3/train/configs/fork/fork_scissors_joint_plate_v4_20ep_full_ft.yaml`，原始權重為 `/workspace/sam3_official/checkpoints/sam3.pt`，輸出目錄為 `/workspace/sam3_official/fork_finetuning/experiments/fork_scissors_joint_plate_v4_20ep_full_ft`。它不會覆蓋上述 5-epoch 續訓或舊版 checkpoint。此版本使用較低學習率，並保留 bfloat16 autocast、停用只適用於 float16 的 gradient scaling；v3 實驗在第 9 個 epoch 發生非有限 matching cost，僅保留作為診斷紀錄，請勿部署其 checkpoint。

worker 預設使用原本的 `/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_balanced_official_full_ft`。新版 `/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_plate_v4_20ep_full_ft` 仍保留；若要再測，可在啟動 worker 時指定 `mm_sam3_vision_tower:=/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_plate_v4_20ep_full_ft`。現有幾何與介入門檻是原本權重的校正值。

> 重要：worker 不能直接載入訓練器產生的 `checkpoint.pt`。必須先執行匯出，讓目錄內包含 `config.json`、processor/tokenizer 設定與 `*.safetensors`。

## 1. 目前使用的檔案與路徑

| 用途 | 路徑 |
| --- | --- |
| 叉子來源資料 | `/workspace/dataset composition/Forks.coco` |
| 剪刀來源資料 | `/workspace/dataset composition/scissors.coco` |
| 聯合資料準備腳本 | `/workspace/sam3_official/fork_finetuning/prepare_joint_fork_scissors_balanced.py` |
| 一鍵資料準備＋訓練腳本 | `/workspace/sam3_official/fork_finetuning/run_fork_scissors_joint_balanced_official_full_ft_train.sh` |
| 訓練設定 | `/workspace/sam3_official/sam3/train/configs/fork/fork_scissors_joint_balanced_official_full_ft.yaml` |
| 整理後的訓練資料 | `/workspace/sam3_official/fork_finetuning/data/ForkScissors.joint.balanced.official.full_ft` |
| Native 訓練權重 | `/workspace/sam3_official/fork_finetuning/experiments/fork_scissors_joint_balanced_official_full_ft/checkpoints/checkpoint.pt` |
| HF 匯出腳本 | `/workspace/sam3_official/fork_finetuning/export_finetuned_hf.py` |
| 目前 worker 預設權重 | `/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_balanced_official_full_ft` |
| Worker launch | `/workspace/ros2_ws/src/vlpoint/launch/worker.launch.py` |
| Worker 程式 | `/workspace/ros2_ws/src/vlpoint/point/serve/model_worker.py` |

目前 worker 預設權重為原本的 epoch 20（train step 560）；新版混合資料集的 epoch 20（train step 1320）仍原樣保留。`finetuning_manifest.json` 記錄來源 checkpoint 與轉換結果，可用來確認 worker 載入的不是原始 `facebook/sam3`。

## 2. Dataset 格式

### 2.1 來源目錄

資料準備腳本會從每個來源資料集的 `train` split 讀取影像與 `_annotations.coco.json`：

```text
/workspace/dataset composition/
├── Forks.coco/
│   └── train/
│       ├── _annotations.coco.json
│       └── <images>
└── scissors.coco/
    └── train/
        ├── _annotations.coco.json
        └── <images>
```

來源 category 名稱可使用：

- 叉子：`fork` 或 `forks`
- 剪刀：`scissor` 或 `scissors`

### 2.2 COCO annotation

每個物件至少需要合法的 COCO bbox；格式是 `[x, y, width, height]`，其中 width、height 必須大於 0。

```json
{
  "images": [
    {"id": 1, "file_name": "example.jpg", "width": 1280, "height": 720}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 80, 220, 500],
      "area": 110000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "fork"}
  ]
}
```

目前是 bbox-only 訓練，沒有使用 instance mask。準備後的 annotation 會統一為：

- `fork`：category ID 1，`noun_phrase: "fork"`
- `scissors`：category ID 2，`noun_phrase: "scissors"`
- `segmentation: []`
- `iscrowd: 0`

準備腳本以 seed 42 做 80/20 的 train/test 切分，影像以 symlink 放入輸出資料夾，不會複製原圖。實際切分、來源 image ID 與數量記錄在：

```text
/workspace/sam3_official/fork_finetuning/data/ForkScissors.joint.balanced.official.full_ft/split_manifest.json
```

目前資料量：

| Split | Fork | Scissors |
| --- | ---: | ---: |
| train images | 28 | 22 |
| train annotations | 190 | 57 |
| test images | 7 | 6 |
| test annotations | 44 | 22 |

雖然兩類原始影像數不同，訓練使用 `CocoConceptBalancedDataset`，每個 epoch 對每一類各抽 14 個 sample，維持 1:1 類別取樣。

## 3. 訓練方法

目前設定是官方 SAM3 image model 的 full fine-tuning：

- 初始化權重：官方 `facebook/sam3`（由 model builder 從 Hugging Face cache 載入）
- prompt：`fork`、`scissors`
- supervision：bbox、GIoU、classification、presence
- segmentation：關閉（`enable_segmentation: false`）
- resolution：1008
- epoch：20
- batch size：1
- gradient accumulation：1
- optimizer：AdamW
- mixed precision：bfloat16
- one-to-many matcher：啟用，`weight=2.0`、`threshold=0.4`、`topk=4`
- 單張 GPU 訓練

因為資料沒有真實 instance mask，請勿直接打開 segmentation loss；若日後加入 polygon/RLE mask，需另外修改 dataset 與 config。

首次使用先安裝官方訓練依賴：

```bash
cd /workspace/sam3_official
python3 -m pip install --no-build-isolation -e '.[train]'
```

執行完整流程：

```bash
bash /workspace/sam3_official/fork_finetuning/run_fork_scissors_joint_balanced_official_full_ft_train.sh
```

此腳本會依序：

1. 合併、切分並驗證叉子與剪刀資料。
2. 檢查 CUDA。
3. 使用官方 trainer 執行 20 epochs 訓練。
4. 將 checkpoint 與 log 寫入 experiment 目錄。

訓練過程可用 TensorBoard 觀察：

```bash
tensorboard --logdir /workspace/sam3_official/fork_finetuning/experiments/fork_scissors_joint_balanced_official_full_ft/tensorboard
```

## 4. 訓練後權重

### 4.1 Native checkpoint

官方 trainer 的輸出位於：

```text
/workspace/sam3_official/fork_finetuning/experiments/fork_scissors_joint_balanced_official_full_ft/checkpoints/checkpoint.pt
```

這是包含 `model`、epoch、steps 等訓練狀態的 native checkpoint，供續訓或轉換使用，**不能直接指定給 Transformers worker**。

### 4.2 匯出成 worker 可載入格式

建議每次匯出到新的版本化目錄，例如：

```bash
cd /workspace/sam3_official/fork_finetuning

python3 export_finetuned_hf.py \
  --checkpoint /workspace/sam3_official/fork_finetuning/experiments/fork_scissors_joint_balanced_official_full_ft/checkpoints/checkpoint.pt \
  --base-model facebook/sam3 \
  --output-dir /workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_YYYYMMDD
```

`export_finetuned_hf.py` 使用 `local_files_only=True` 載入 base model，因此執行匯出前，本機 Hugging Face cache 必須已存在 `facebook/sam3`。

匯出腳本會把微調後的 detector 權重覆蓋到本機的 HF base model，檢查 key/shape 是否一致，並驗證權重確實與 base model 不同。因目前是 bbox-only 訓練，mask decoder 會保留 base model 權重。

成功的輸出目錄至少應包含：

```text
config.json
finetuning_manifest.json
model.safetensors.index.json
model-00001-of-*.safetensors
processor_config.json
tokenizer.json
tokenizer_config.json
```

確認 manifest 的來源、epoch 與權重變更數：

```bash
python3 -m json.tool \
  /workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_YYYYMMDD/finetuning_manifest.json
```

其中應確認：

- `fine_tuned_checkpoint` 指向本次訓練的 `checkpoint.pt`
- `checkpoint_epoch` 是預期 epoch
- `changed_tensor_count` 大於 0
- `shape_mismatch_count` 與 `unexpected_key_count` 都是 0

也可直接做一次離線載入檢查：

```bash
python3 -c "from transformers import Sam3Model, Sam3Processor; p='/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_YYYYMMDD'; Sam3Processor.from_pretrained(p, local_files_only=True); Sam3Model.from_pretrained(p, local_files_only=True); print('HF export load OK')"
```

## 5. 替換到 worker

SAM3 權重是在 worker 啟動時由 `--mm-sam3-vision-tower` 指定。SAM3 detection 沒有另外指定 model path 時，也會沿用這個路徑。

### 5.1 建議方式：啟動時覆寫路徑

先停止舊 worker，再用新匯出目錄啟動：

```bash
cd /workspace/ros2_ws
source install/setup.bash

ros2 launch vlpoint worker.launch.py \
  mm_sam3_vision_tower:=/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_YYYYMMDD
```

這種方式不需修改程式，也保留舊版權重供回滾。不要在 worker 還在執行時覆蓋其權重檔；換版本後必須重啟 worker。

### 5.2 固定成預設版本

若希望之後手動執行 `ros2 launch vlpoint worker.launch.py` 時自動使用新權重，更新以下兩處的預設路徑：

1. `/workspace/ros2_ws/src/vlpoint/launch/worker.launch.py` 的 `mm_sam3_vision_tower` default value。
2. `/workspace/ros2_ws/src/vlpoint/point/serve/model_worker.py` 的 `--mm-sam3-vision-tower` default，以及 detector fallback path。

若 workspace 使用 symlink install，修改後可直接重啟；否則重新 build 並 source：

```bash
cd /workspace/ros2_ws
colcon build --packages-select vlpoint
source install/setup.bash
```

### 5.3 啟動後確認

確認實際 worker command line 含有正確路徑：

```bash
ps -ef | grep '[p]oint.serve.model_worker'
```

另外核對該目錄的 `finetuning_manifest.json`。只看到 `facebook/sam3` 或把 `checkpoint.pt` 直接放到參數中，都不代表已正確載入微調權重。

## 6. 目前正式使用版本

目前 worker launch 與程式預設皆指向：

```text
/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_balanced_official_full_ft
```

此版本的 manifest 顯示：

- checkpoint epoch：20
- train steps：560
- changed tensors：1412
- changed parameters：830,602,293
- shape mismatch：0
- unexpected keys：0
- mask decoder updated：false（符合 bbox-only 訓練設定）

新版權重保留在 `/workspace/sam3_official/fork_finetuning/exported_models/sam3_fork_scissors_joint_plate_v4_20ep_full_ft`，可用 `mm_sam3_vision_tower` 明確指定測試。`SAM3_CALIBRATED_PROFILE` 與目前的原本權重相符。

## 7. 注意事項

- worker 的 query threshold、IoM、幾何限制與介入條件屬於 runtime calibration，位於 `model_worker.py`，不會被存入 SAM3 權重。
- 更換權重不會自動更換上述 runtime calibration；若新模型分數分布改變，需另外重新校正。
- 保留每個匯出版本的 `finetuning_manifest.json`，才能追蹤 worker 使用的 checkpoint、epoch 與轉換結果。
- 若新增資料後要重跑相同 experiment，先備份舊的 experiment 與 exported model，避免失去可回滾版本。

## 8. Upstream training compatibility

目前 upstream SAM3.1 的 `Mlp` 在 autograd 開啟時仍可能呼叫 inference-only fused `addmm_act` kernel。此 checkout 在需要梯度時使用標準 `fc1 -> activation` 路徑，推論時維持 fused path；不會改變模型架構或 state-dict 格式。
