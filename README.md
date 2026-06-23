# SAM2-UNet Fusion

基于 SAM2 视觉编码器与 U-Net 解码器的多任务分割框架，通过可切换的特征桥接模块（FeatureBridge）融合 DarkIR 空间/频域增强与 ParameterNet 多专家动态投影。

支持多个分割基准数据集，通过配置文件切换任务。

## 架构概览

```
输入图像
    │
    ▼
┌─────────────────────┐
│  SAM2 Hiera-Large   │  冻结编码器 + Adapter 微调
│  (4 级特征输出)      │
└───┬───┬───┬───┬─────┘
    │   │   │   │
    ▼   ▼   ▼   ▼      ← FeatureBridge（5 种消融模式可选）
┌─────────────────────┐
│  U-Net 解码器        │  三级深度监督输出
│  (3 级分割输出)      │
└───┬───┬───┬──────────┘
    │   │   │
    ▼   ▼   ▼
  P1   P2  P3
```

**五种特征桥接模式**：

| 模式           | 投影方式               | DarkIR 增强 | 说明                   |
| -------------- | ---------------------- | ----------- | ---------------------- |
| `rfb`          | 基线 RFB               | 否          | 原 SAM2-UNet 基线实现  |
| `static`       | 普通 1×1 Conv          | 否          | 最简投影基线           |
| `parameternet` | 多专家动态投影         | 否          | ParameterNet 路由机制  |
| `darkir`       | 普通 1×1 Conv          | 是          | DarkIR 空间+频域增强   |
| `full`         | 多专家动态投影         | 是          | 完整融合方案           |

## 支持的数据集

### 息肉分割（Polyp Segmentation）

| 数据集         | 角色     | 规模            | 配置键         | 状态     |
| -------------- | -------- | --------------- | -------------- | -------- |
| Kvasir-SEG     | 训练+测试 | 1000 / 100     | `Kvasir`       | ✅ 已实现 |
| CVC-ClinicDB   | 训练+测试 | 612 / 62       | `CVC-ClinicDB` | ✅ 已实现 |
| CVC-300        | 仅测试   | 60              | `CVC-300`      | ✅ 已实现 |
| CVC-ColonDB    | 仅测试   | 380             | `CVC-ColonDB`  | ✅ 已实现 |
| ETIS-Larib     | 仅测试   | 196             | `ETIS`         | ✅ 已实现 |

### 伪装目标检测（Camouflaged Object Detection）

| 数据集         | 角色     | 规模            | 配置键         | 状态     |
| -------------- | -------- | --------------- | -------------- | -------- |
| CAMO           | 训练+测试 | 1000 / 250     | `CAMO`         | ✅ 已实现 |
| COD10K         | 训练+测试 | 3040 / 2026    | `COD10K`       | ✅ 已实现 |
| NC4K           | 仅测试   | 4121            | `NC4K`         | ✅ 已实现 |
| CHAMELEON      | 仅测试   | 76              | `CHAMELEON`    | ✅ 已实现 |

COD 数据已准备并校验；对应的数据加载器和训练配置已实现，可直接通过 `main.py` 使用。

通过 `--config` 参数切换不同任务的数据集和训练配置。每个配置文件独立定义数据路径、输入尺寸、训练超参和评估指标。跨数据集测试（在未见过的数据集上评估）用于衡量模型的泛化能力。

## 项目结构

```
qzx/
├── main.py                          # 统一 CLI 入口
├── configs/
│   ├── polyp_train.json             # 息肉分割训练配置
│   ├── polyp_sources.json           # 息肉数据集来源与校验
│   ├── cod_train.json               # 伪装目标检测训练配置
│   ├── cod_sources.json             # COD 数据集来源与校验
│   └── sam2/                        # SAM2 模型配置
├── src/sam2unet/
│   ├── __init__.py                  # 懒加载导出模型类
│   ├── runtime.py                   # SAM2 构建与输入校验
│   ├── baseline.py                  # 原 SAM2-UNet（RFB 桥接）
│   ├── experimental_darkir.py       # 实验变体（DBlock_DAT + FusedEnhanceBlock）
│   ├── fusion.py                    # 核心融合模型（5 种消融模式）
│   ├── training.py                  # 损失函数、指标、训练/评估循环、checkpoint I/O
│   ├── polyp_dataset.py             # 息肉数据集（CSV manifest）
│   ├── cod_dataset.py               # 伪装目标数据集
│   ├── data.py                      # 数据准备 CLI
│   ├── experiment.py                # 实验 CLI（smoke / train / evaluate）
│   └── monitor.py                   # 实时训练监控（TensorBoard + tqdm）
├── tests/                           # 测试套件
├── scripts/
│   ├── download_polyp_data.ps1      # 下载息肉数据集
│   └── prepare_local_polyp_data.ps1 # 从完整归档准备息肉数据
├── checkpoints/
│   └── sam2_hiera_large.pt          # 预训练 SAM2 权重（~898 MB）
├── runs/                            # 训练输出（按任务/模式组织）
└── docs/                            # 论文草稿与设计文档
```

## 环境配置

Python ≥ 3.10，PyTorch ≥ 2.1，推荐使用 conda 管理环境：

```powershell
# 创建并激活环境
conda create -n sam2 python=3.10 -y
conda activate sam2

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision

# 安装项目依赖
pip install -r requirements.txt
```

构建真实 Hiera-L 编码器需要 Meta 官方 SAM2（Windows 推荐 WSL）：

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

## 数据准备

数据统一放在 `data/` 下，按任务组织：

```text
data/
├── polyps/                          # 息肉分割
│   ├── archives/                    # 原始压缩包
│   └── prepared/                    # 标准化后的数据
│       ├── train/                   # 训练集
│       └── test/                    # 各测试集
│           ├── Kvasir/
│           ├── CVC-ClinicDB/
│           ├── CVC-300/
│           ├── CVC-ColonDB/
│           └── ETIS/
└── cod/                             # 伪装目标检测
    ├── archives/
    └── prepared/
        ├── train/
        │   ├── CAMO/
        │   └── COD10K/
        └── test/
            ├── CAMO/
            ├── COD10K/
            ├── NC4K/
            └── CHAMELEON/
```

### 息肉分割

**方式一：下载 PraNet 标准划分**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_polyp_data.ps1
```

**方式二：从完整归档准备**

将完整数据集放在 `data/polyps/archives/` 下：

```text
data/polyps/archives/PraNet-TrainDataset.zip
data/polyps/archives/kvasir-seg.zip
data/polyps/archives/CVC-ClinicDB.zip
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_local_polyp_data.ps1
```

方式二只从上述三个归档重建训练集、Kvasir 和 CVC-ClinicDB；扩展测试集需要使用 PraNet 测试集来源另行补齐。

完整准备后数据位于 `data/polyps/prepared/`：

```text
train/                 1450 对（图像 + 掩码）
test/Kvasir            100 对
test/CVC-ClinicDB       62 对
test/CVC-300            60 对
test/CVC-ColonDB        380 对
test/ETIS               196 对
```

校验数据完整性：

```powershell
python -m sam2unet.data validate data/polyps/prepared
```

**扩展测试集**：

准备额外的息肉测试集用于跨数据集泛化评估：

| 数据集      | 规模 | 说明                              |
| ----------- | ---- | --------------------------------- |
| CVC-300     | 60   | 结肠镜图像，息肉边界清晰          |
| CVC-ColonDB | 380  | 结肠镜数据库，包含小息肉          |
| ETIS-Larib  | 196  | 高分辨率内窥镜图像                |

### 伪装目标检测（已实现）

训练集：CAMO + COD10K；测试集：CAMO、COD10K、NC4K、CHAMELEON。

数据位于 `data/cod/prepared/`，来源与校验记录见 `configs/cod_sources.json`。数据加载器 `src/sam2unet/cod_dataset.py` 与训练配置 `configs/cod_train.json` 已实现，可通过 `main.py` 的 `smoke`、`train` 与 `evaluate` 模式直接使用。

| 数据集  | 规模   | 说明                                |
| ------- | ------ | ----------------------------------- |
| CAMO    | 1250   | 1000 训练 + 250 测试，伪装物体分割  |
| COD10K  | 5066   | 3040 训练 + 2026 测试，大规模 COD   |
| NC4K    | 4121   | 纯测试集，自然场景伪装目标          |
| CHAMELEON | 76   | 纯测试集，经典伪装目标分割基准      |

当前实现按二值分割训练与评估。NC4K 会评估其二值掩码；原始实例/annotations 文件仍保留在准备后的数据目录中，但当前二值损失与指标不会使用这些实例级标注。
数据加载器会分别将图像（双线性）与二值掩码（最近邻）归一化到 `input_size`；这是为兼容上游 CHAMELEON 官方数据中的两个源尺寸不一致样本，并非下载失败。

## 使用方法

`main.py` 是统一运行入口，默认执行 smoke 小规模测试：

```powershell
python main.py --help
```

### Smoke 测试

快速验证环境和模型前向传播：

```powershell
python main.py
# 或显式指定
python main.py --mode smoke

# COD 真实数据 smoke
python main.py --mode smoke --config configs/cod_train.json --device cpu
```

### 正式训练

通过 `--config` 选择任务，`--bridge-mode` 选择消融模式：

```powershell
# 息肉分割 — 完整融合模型（训练+跨数据集测试）
python main.py --mode train --config configs/polyp_train.json --bridge-mode full --epochs 20

# 伪装目标检测
python main.py --mode train --config configs/cod_train.json --bridge-mode full --epochs 30

# 消融实验：逐个训练各桥接模式
python main.py --mode train --config configs/polyp_train.json --bridge-mode rfb --epochs 20
python main.py --mode train --config configs/polyp_train.json --bridge-mode static --epochs 20
python main.py --mode train --config configs/polyp_train.json --bridge-mode parameternet --epochs 20
python main.py --mode train --config configs/polyp_train.json --bridge-mode darkir --epochs 20
```

训练过程中模型会在配置文件中定义的所有测试集上自动评估，`history.json` 记录每个测试集的 Dice、IoU、MAE 指标，便于分析跨数据集泛化能力。

从 checkpoint 恢复训练：

```powershell
python main.py --mode train --config configs/polyp_train.json --bridge-mode full --epochs 20 --resume runs/polyps/full/latest.pt
```

### 评估

```powershell
python main.py --mode evaluate --config configs/polyp_train.json --bridge-mode full --checkpoint runs/polyps/full/best.pt

# COD 评估
python main.py --mode evaluate --config configs/cod_train.json --bridge-mode full --checkpoint runs/cod/full/best.pt
```

### 常用参数

| 参数              | 说明                                      | 默认值          |
| ----------------- | ----------------------------------------- | --------------- |
| `--mode`          | 运行模式：`smoke` / `train` / `evaluate` | `smoke`         |
| `--config`        | 任务配置文件                              | `configs/polyp_train.json` |
| `--bridge-mode`   | 特征桥接消融模式                          | 配置文件最后一项 |
| `--device`        | 运行设备：`auto` / `cpu` / `cuda` / `cuda:N` | `auto`      |
| `--epochs`        | 训练轮数（仅 train 模式）                 | 配置文件值      |
| `--resume`        | 恢复训练的 checkpoint 路径（仅 train）    | 无              |
| `--checkpoint`    | 评估用 checkpoint 路径（仅 evaluate）     | 必填            |
| `--limit-train`   | 限制训练样本数（诊断用）                  | 无              |
| `--limit-test`    | 限制测试样本数（诊断用）                  | 无              |
| `--no-monitor`    | 禁用实时监控（仅 train 模式）             | 启用            |

## 实时训练监控

训练时自动启用 TensorBoard 实时监控和 tqdm 终端进度条。

### 终端进度条

```
Epoch 1:  45%|████▌      | 540/1200 [02:15<02:46, loss=0.3421, lr=9.87e-04]
Epoch 1 summary — loss=0.3652 | Kvasir: Dice=0.8234 IoU=0.7102 | CVC-ClinicDB: Dice=0.7891 IoU=0.6654
```

### TensorBoard

```powershell
# 查看所有任务的训练曲线
tensorboard --logdir=runs/

# 仅查看特定任务
tensorboard --logdir=runs/polyps/
```

浏览器访问 `http://localhost:6006` 查看：

- `train/batch_loss` — 每个 batch 的 loss 曲线
- `train/epoch_loss` — 每个 epoch 的平均 loss
- `val/{测试集}/dice` — 各测试集 Dice / IoU / MAE
- `val/average_dice` — 综合 Dice 对比
- `train/learning_rate` — 学习率调度曲线

TensorBoard 日志保存在 `runs/<任务>/<bridge_mode>/tb_logs/`。

禁用监控：

```powershell
python main.py --mode train --config configs/polyp_train.json --bridge-mode full --no-monitor
```

## 训练配置

每个任务的配置文件独立管理。以息肉分割为例（`configs/polyp_train.json`）：

```json
{
  "task": "binary_segmentation",
  "input_size": [352, 352],
  "batch_size": 1,
  "gradient_accumulation_steps": 12,
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "epochs": 20,
  "num_experts": 4,
  "bridge_modes": ["rfb", "static", "full"]
}
```

关键设计：
- **Batch size 1 + 梯度累积 12 步** = 有效 batch size 12，适配小显存
- **自动混合精度（AMP）**：CUDA 上自动启用，节省显存
- **梯度裁剪**：最大梯度范数 1.0，稳定训练
- **Cosine Annealing**：学习率调度

添加新任务时，复制现有配置文件并修改数据路径、输入尺寸和评估指标即可。

## Checkpoint 管理

训练产物按 `runs/<任务>/<bridge_mode>/` 组织：

```text
runs/
├── polyps/
│   ├── full/
│   │   ├── latest.pt              # 最新 checkpoint
│   │   ├── best.pt                # 最佳 Dice checkpoint
│   │   ├── history.json           # 训练指标记录（含所有测试集）
│   │   ├── training_summary.json  # 训练总结
│   │   └── tb_logs/               # TensorBoard 日志
│   ├── rfb/
│   ├── static/
│   ├── parameternet/
│   └── darkir/
└── cod/
    ├── full/
    ├── rfb/
    └── ...
```

每个 checkpoint 的 `history.json` 记录所有测试集的指标，便于分析跨数据集泛化性能。

编程方式加载 checkpoint：

```python
from sam2unet import SAM2UNetFusion

# 从 checkpoint 恢复完整训练状态
model = SAM2UNetFusion.from_checkpoint("runs/polyps/full/best.pt")

# 从原 SAM2-UNet checkpoint 初始化公共层
model.load_baseline_checkpoint("baseline.pt")
```

## 路由监控

开启路由统计可观察 ParameterNet 多专家路由的权重分布：

```python
model = SAM2UNetFusion(return_router_stats=True)
outputs, router_stats = model(images)

# router_stats 包含每级桥接的：
# - expert_weights: 各专家权重
# - mean_usage: 专家平均使用率
# - routing_entropy: 平均路由熵
```

DarkIR 空间与频域分支可通过 `enable_spatial` 和 `enable_frequency` 独立消融。

## 测试

```powershell
python -m pytest tests/ -q
```

测试覆盖：
- 模型架构：5 种桥接模式、前向传播、checkpoint 往返
- 训练组件：损失函数、指标计算、训练循环、checkpoint I/O
- 数据管线：数据集加载、manifest 校验、数据准备 CLI
- 运行时：SAM2 构建、输入校验、设备放置
- 实验 CLI：smoke 测试、参数解析、产物生成
- 监控模块：TensorBoard 集成、tqdm 进度条

## 许可

本项目为学术研究用途。SAM2 编码器来自 [Meta SAM2](https://github.com/facebookresearch/sam2)，遵循其原始许可。
