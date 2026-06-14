# SAM2-UNet DarkIR + ParameterNet 稳健融合版

本项目包含三个 SAM2-UNet 模型变体，并采用标准 `src` Python 包布局：

```text
qzx/
├─ src/sam2unet/
│  ├─ baseline.py
│  ├─ experimental_darkir.py
│  ├─ fusion.py
│  └─ runtime.py
├─ tests/
├─ docs/
│  ├─ design/
│  ├─ analysis/
│  ├─ papers/
│  └─ superpowers/plans/
├─ third_party/sam2/
├─ checkpoints/
├─ pyproject.toml
└─ requirements.txt
```

## 模型

- `sam2unet.baseline.BaselineSAM2UNet`：论文版 SAM2-UNet 基线。
- `sam2unet.ExperimentalDarkIRSAM2UNet`：DBlock_DAT 与 FusedEnhanceBlock 实验版本。
- `sam2unet.SAM2UNetFusion`：ParameterNet 动态投影与 DarkIR 空间/频域增强融合版本。

## 安装

项目会自动使用 `third_party/sam2/` 中的 Meta SAM2 源码，也支持环境中已安装的 SAM2。

```powershell
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python -m pytest -q
```

当前项目已在 Windows、Python 3.9、PyTorch 2.1.1、CUDA 11.8 上完成真实 Hiera-L 验证。

## 运行

运行基线与实验模型：

```powershell
python -m sam2unet.baseline
python -m sam2unet.experimental_darkir
```

使用融合模型：

```python
from sam2unet import SAM2UNetFusion

model = SAM2UNetFusion(
    checkpoint_path="checkpoints/sam2_hiera_large.pt",
    bridge_mode="full",
    num_experts=4,
    sam_device="cuda",
)
out, out1, out2 = model(images)
```

`sam_device` 默认为 `"cpu"`，可设为 `"cuda"` 将完整模型构建到 GPU。输入必须是 BCHW 四维张量，且高度和宽度均能被 32 整除；设计目标尺寸为 `352x352`。

支持的 `bridge_mode`：

| 模式 | 投影 | DarkIR 增强 |
|---|---|---|
| `rfb` | 原基线 RFB 的兼容实现 | 否 |
| `static` | 普通 `1x1 Conv` | 否 |
| `parameternet` | 多专家动态投影 | 否 |
| `darkir` | 普通 `1x1 Conv` | 是 |
| `full` | 多专家动态投影 | 是 |

开启路由监控：

```python
model = SAM2UNetFusion(return_router_stats=True)
(out, out1, out2), router_stats = model(images)
```

DarkIR 空间与频域分支可通过 `enable_spatial` 和 `enable_frequency` 独立消融。

## Checkpoint

官方 Hiera-L 权重位于：

```text
checkpoints/sam2_hiera_large.pt
```

融合模型 checkpoint 同时保存配置与权重：

```python
model.save_checkpoint("fusion.pt")
restored = SAM2UNetFusion.from_checkpoint("fusion.pt")
```

从原 SAM2-UNet checkpoint 初始化公共部分：

```python
model.load_baseline_checkpoint("baseline.pt")
```

## 文档

- 设计文档：`docs/design/`
- 关系分析：`docs/analysis/`
- 论文原文：`docs/papers/`

当前目录没有任务数据集、数据划分或训练脚本，因此无法生成可信的 mIoU、Dice 和 MAE 结论。

## 已验证范围

- 三种模型均可加载官方 `sam2_hiera_large.pt`。
- 三种模型均通过 CPU 直接入口、CUDA 前向、CUDA 反向和 CUDA 混合精度前向。
- `352x352`、batch size 1 的 CUDA 反向峰值保留显存约为 `2.0-2.1 GB`。
- 测试命令：`python -m pytest -q`。
