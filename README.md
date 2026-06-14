# SAM2-UNet DarkIR + ParameterNet 稳健融合版

本目录在保留 `SAM2UNet.py` 与 `SAM2UNet_dblock_dat_fused_rfbhou.py` 不变的前提下，新增了 `SAM2UNet_darkir_parameternet.py`。新模型将 ParameterNet 风格的多专家动态 `1x1` 投影与 DarkIR 启发的空间/频域特征增强放入统一 `FeatureBridge`，并保留 SAM2-UNet 的三级 U-Net 解码器与三个分割输出。

## 文件

- `SAM2UNet_darkir_parameternet.py`：融合模块与完整模型。
- `tests/test_darkir_parameternet_blocks.py`：动态投影、DarkIR 增强与桥接层测试。
- `tests/test_sam2unet_fusion.py`：五种消融模式、整模输出、路由统计和 checkpoint 测试。
- `SAM2UNet融合DarkIR与ParameterNet稳健版设计文档.md`：架构与实验设计依据。

## 环境

块级测试只依赖 PyTorch 与 pytest：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

构建真实 Hiera-L 编码器还需要 Meta 官方 SAM2。当前官方 SAM2 要求 Python 3.10+、PyTorch 2.5.1+，Windows 推荐使用 WSL：

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

本项目默认使用官方原始 SAM2 Hiera-L 配置 `configs/sam2/sam2_hiera_l.yaml`。使用 SAM2.1 时，应显式传入：

```python
model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
```

## 使用

```python
from SAM2UNet_darkir_parameternet import SAM2UNetFusion

model = SAM2UNetFusion(
    checkpoint_path="checkpoints/sam2_hiera_large.pt",
    bridge_mode="full",
    num_experts=4,
)
out, out1, out2 = model(images)
```

支持的 `bridge_mode`：

| 模式             | 投影                  | DarkIR 增强 |
| ---------------- | --------------------- | ----------- |
| `rfb`          | 原基线 RFB 的兼容实现 | 否          |
| `static`       | 普通 `1x1 Conv`     | 否          |
| `parameternet` | 多专家动态投影        | 否          |
| `darkir`       | 普通 `1x1 Conv`     | 是          |
| `full`         | 多专家动态投影        | 是          |

开启路由监控：

```python
model = SAM2UNetFusion(return_router_stats=True)
(out, out1, out2), router_stats = model(images)
```

每级路由统计包含样本权重、专家平均使用率和平均路由熵。DarkIR 空间与频域分支可通过 `enable_spatial` 和 `enable_frequency` 独立消融。

## Checkpoint

融合模型 checkpoint 同时保存模型配置与权重：

```python
model.save_checkpoint("fusion.pt")
restored = SAM2UNetFusion.from_checkpoint("fusion.pt")
```

从原 SAM2-UNet checkpoint 初始化公共编码器、Adapter、解码器和输出头时使用非严格加载：

```python
model.load_baseline_checkpoint("baseline.pt")
```

## 训练边界

本地息肉数据已经准备在 Git 忽略的 `data/polyps/prepared/` 目录中，训练与评估入口也已补齐。可信的 mIoU、Dice、MAE、延迟和显存结论仍需完成正式训练后生成。训练时应继续使用设计文档中的分割损失、梯度裁剪与五模式消融顺序，并在同一数据划分和随机种子集合下比较。

## Polyp training and evaluation

The project includes a complete manifest-based training and evaluation CLI.
The default configuration is conservative for the local 4 GB GPU: batch size
1, automatic mixed precision on CUDA, gradient accumulation, and gradient
clipping.

根目录的 `main.py` 是推荐的统一运行入口。直接运行时默认执行小规模
`smoke` 测试，避免误启动正式训练：

```powershell
python main.py
```

正式训练完整融合模型：

```powershell
python main.py --mode train --bridge-mode full --epochs 20
```

从最近的 checkpoint 继续训练：

```powershell
python main.py --mode train --bridge-mode full --epochs 20 `
  --resume runs/polyps/full/latest.pt
```

评估最佳 checkpoint：

```powershell
python main.py --mode evaluate --bridge-mode full `
  --checkpoint runs/polyps/full/best.pt
```

可使用 `--device cpu|cuda|cuda:N`、`--limit-train` 和 `--limit-test`
控制运行设备与诊断数据量。查看全部参数：

```powershell
python main.py --help
```

Run the complete lightweight workflow smoke test:

```powershell
python -m sam2unet.experiment smoke --config configs/polyp_train.json
```

Train one bridge mode:

```powershell
python -m sam2unet.experiment train `
  --config configs/polyp_train.json `
  --bridge-mode static
```

Resume training:

```powershell
python -m sam2unet.experiment train `
  --config configs/polyp_train.json `
  --bridge-mode static `
  --resume runs/polyps/static/latest.pt
```

Evaluate a training checkpoint:

```powershell
python -m sam2unet.experiment evaluate `
  --config configs/polyp_train.json `
  --bridge-mode static `
  --checkpoint runs/polyps/static/best.pt
```

Use `--limit-train` and `--limit-test` for short diagnostic runs. Training
artifacts are written under `runs/polyps/` and include `latest.pt`, `best.pt`,
`history.json`, and evaluation summaries.

## Polyp dataset preparation

Download and prepare the official PraNet polyp split:

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/download_polyp_data.ps1
```

The command keeps only the normalized data required by this project:

```text
data/polyps/prepared/train                 1450 pairs
data/polyps/prepared/test/Kvasir            100 pairs
data/polyps/prepared/test/CVC-ClinicDB        62 pairs
```

Validate an existing prepared dataset:

```powershell
$env:PYTHONPATH = "src"
python -m sam2unet.data validate data/polyps/prepared
```

The initial experiment settings are recorded in `configs/polyp_train.json`.
Archive origins and SHA-256 checksums are recorded in
`configs/polyp_sources.json`.

When the complete Kvasir-SEG and CVC-ClinicDB archives are already available,
place them beside `PraNet-TrainDataset.zip` under `data/polyps/archives/`:

```text
data/polyps/archives/PraNet-TrainDataset.zip
data/polyps/archives/kvasir-seg.zip
data/polyps/archives/CVC-ClinicDB.zip
```

Then reproduce the exact PraNet split by using the complete datasets as the
source of the test complements:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_local_polyp_data.ps1
```
