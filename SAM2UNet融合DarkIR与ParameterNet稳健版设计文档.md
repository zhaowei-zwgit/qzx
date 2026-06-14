# SAM2-UNet 融合 DarkIR 与 ParameterNet 稳健版设计文档

## 1. 文档目标

本设计旨在将以下两类创新稳定地引入 SAM2-UNet：

- **DarkIR**：利用空间大感受野和频域幅度信息增强特征。
- **ParameterNet**：通过输入相关的多专家动态卷积，在较小 FLOPs 增量下提高模型参数容量。

设计遵循以下原则：

1. 保留 `SAM2UNet.py` 作为不可变基线，确保实验结果可复现。
2. 保留 `SAM2UNet_dblock_dat_fused_rfbhou.py` 作为已有实验版本，不继续叠加新变量。
3. 新建独立融合模型，避免修改冻结的 SAM2 Hiera 主干。
4. DarkIR 与 ParameterNet 分别作用于特征增强和动态投影，保证可以独立消融。
5. 不使用当前实验文件中的 `DynamicAdaptiveTanh`，避免引入无法归因于三篇论文的额外变量。

---

## 2. 可行性结论

DarkIR 与 ParameterNet 可以共同用于 SAM2-UNet，但二者应解决不同层次的问题：

| 方法 | 原论文核心创新 | 在新模型中的作用 |
|---|---|---|
| SAM2-UNet | 冻结 SAM2 Hiera 编码器，使用 Adapter、RFB 和 U-Net 解码器完成分割 | 提供基础模型骨架 |
| ParameterNet | 使用输入相关的多专家动态卷积，在相近卷积 FLOPs 下增加参数容量 | 动态完成四级编码特征的通道投影 |
| DarkIR | 使用频域 FreMLP 和大感受野空间注意力处理退化图像 | 增强投影后的多尺度分割特征 |

推荐的融合关系为：

```text
SAM2 Hiera-L 编码器
        ↓ 四级编码特征
ParameterNet 动态投影
        ↓ 统一为 64 通道
DarkIR 空间/频域特征增强
        ↓
原 SAM2-UNet U-Net 解码器
        ↓
主输出 + 两个辅助输出
```

该方案不尝试复现完整 DarkIR 图像恢复网络，也不将 ParameterNet 扩展到整个 SAM2 主干。这样可以限制改动范围，降低训练风险，并使性能变化可以被清楚归因。

---

## 3. 文件组织

### 3.1 保持不变的文件

- `SAM2UNet.py`  
  原始论文基线。不得加入 DarkIR 或 ParameterNet 模块。

- `SAM2UNet_dblock_dat_fused_rfbhou.py`  
  当前 DBlock_DAT 与 FusedEnhanceBlock 实验版本。保留用于横向比较，不继续修改。

### 3.2 建议新增的文件

| 文件 | 职责 |
|---|---|
| `SAM2UNet_darkir_parameternet.py` | 定义动态投影、DarkIR 增强块、特征桥接层和完整融合模型 |
| `tests/test_darkir_parameternet_blocks.py` | 测试动态卷积、路由、频域分支和残差初始化 |
| `tests/test_sam2unet_fusion.py` | 测试不同消融模式和完整模型输出接口 |

新增文件可以复用 `SAM2UNet.py` 中的 `Adapter`、`Up` 和 `RFB_modified`，但不能改变这些基线类的行为。

---

## 4. 总体架构

### 4.1 数据流

```text
输入图像 [B, 3, H, W]
    ↓
冻结的 SAM2 Hiera-L + 可训练 Adapter
    ├─ x1 [B, 144, H/4,  W/4]
    ├─ x2 [B, 288, H/8,  W/8]
    ├─ x3 [B, 576, H/16, W/16]
    └─ x4 [B,1152, H/32, W/32]
    ↓
四个 FeatureBridge
    ├─ ParameterNetDynamicProjection: Ci → 64
    └─ DarkIRFeatureEnhancer: 64 → 64
    ↓
增强特征 f1、f2、f3、f4
    ↓
原 SAM2-UNet 三级 U-Net 解码器
    ↓
out、out1、out2
```

### 4.2 模型接口

建议完整模型接口为：

```python
class SAM2UNetFusion(nn.Module):
    def __init__(
        self,
        checkpoint_path=None,
        bridge_mode="full",
        num_experts=4,
        return_router_stats=False,
    ):
        ...

    def forward(self, x):
        ...
```

默认输出接口必须与基线兼容：

```python
out, out1, out2 = model(x)
```

仅当 `return_router_stats=True` 时额外返回路由统计：

```python
(out, out1, out2), router_stats = model(x)
```

---

## 5. ParameterNet 动态投影设计

### 5.1 使用位置

ParameterNet 仅替换原 SAM2-UNet 中四个 RFB 的“通道压缩”职责：

```text
144  → 64
288  → 64
576  → 64
1152 → 64
```

不将动态卷积加入 Hiera block、Adapter 或 U-Net 解码器。原因是这些位置会显著扩大改动范围，使预训练权重利用、训练稳定性和性能归因变得困难。

### 5.2 动态卷积公式

对输入特征 `X`，每个动态投影包含 `M` 个 `1×1` 卷积专家：

```text
W1, W2, ..., WM
```

首先根据输入生成路由权重：

```text
α = Softmax(MLP(GlobalAveragePool(X)))
```

然后融合专家权重：

```text
W_dynamic = Σ αi Wi
```

最终只执行一次动态 `1×1` 卷积：

```text
Y = Conv(X, W_dynamic)
```

这与“分别执行 M 次卷积后再融合输出”不同。后者会把主要卷积 FLOPs 放大 M 倍，不符合 ParameterNet 的设计目标。

### 5.3 建议接口

```python
class ParameterNetDynamicProjection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=64,
        num_experts=4,
        router_reduction=16,
    ):
        ...

    def forward(self, x, return_router=False):
        ...
```

### 5.4 推荐实现方式

路由器结构：

```text
AdaptiveAvgPool2d(1)
→ Flatten
→ Linear(Cin, max(Cin / 16, 16))
→ GELU
→ Linear(hidden, M)
→ Softmax
```

对于 batch 内每个样本，使用不同的融合卷积核。建议通过 batch-grouped convolution 一次完成：

1. 将动态融合后的权重整理为 `[B × Cout, Cin, 1, 1]`。
2. 将输入整理为 `[1, B × Cin, H, W]`。
3. 调用 `F.conv2d(..., groups=B)`。
4. 将结果恢复为 `[B, Cout, H, W]`。

这种实现保留单次主卷积的理论 FLOPs，但需要实际测量 GPU 延迟，因为 grouped convolution 的真实运行效率依赖硬件和 batch size。

### 5.5 初始化策略

为避免训练初期路由不稳定：

- 路由器最后一层的权重和偏置初始化为 0，使初始路由接近均匀分配。
- 第一个卷积专家使用标准 Kaiming 初始化。
- 其他专家从第一个专家复制，并加入极小随机扰动，避免专家长期完全对称。
- 默认专家数量为 `M=4`。

### 5.6 ParameterNet 模块不负责的内容

- 不负责频域增强。
- 不负责扩大空间感受野。
- 不改变 Hiera 主干权重。
- 不引入稀疏 top-k MoE；稳健版使用连续 Softmax 混合。

---

## 6. DarkIR 特征增强设计

### 6.1 使用位置

DarkIR 增强块只处理已经统一为 64 通道的四级特征：

```text
ParameterNetDynamicProjection → DarkIRFeatureEnhancer
```

统一通道后再增强具有以下优势：

- 四个尺度可以使用相同模块结构。
- 参数量和显存更可控。
- 不需要直接处理 1152 通道的高层特征。
- 便于独立比较静态投影和动态投影。

### 6.2 模块结构

```python
class DarkIRFeatureEnhancer(nn.Module):
    def __init__(
        self,
        channels=64,
        dw_expand=2,
        ffn_expand=2,
        dilations=(1, 4, 9),
    ):
        ...

    def forward(self, x):
        ...
```

模块包含两个可独立消融的分支：

1. 空间大感受野分支；
2. 频域幅度增强分支。

### 6.3 空间分支

空间分支借鉴 DarkIR DBlock 的 Di-SpAM：

```text
LayerNorm2d
→ 1×1 Conv 扩展通道
→ 三个并行 depth-wise 3×3 Conv
   ├─ dilation=1
   ├─ dilation=4
   └─ dilation=9
→ 分支求和
→ SimpleGate
→ Simplified Channel Attention
→ 1×1 Conv 恢复为 64 通道
→ 零初始化残差缩放
```

空间分支输出：

```text
y = x + β × SpatialBranch(x)
```

其中 `β` 初始化为 0，使模块初始行为接近恒等映射。

### 6.4 频域分支

频域分支借鉴 DarkIR FreMLP，只修改 FFT 幅度并保留相位：

```text
LayerNorm2d
→ FFT
→ 分离 magnitude 与 phase
→ 使用 1×1 Conv MLP 处理 magnitude
→ magnitude 与原 phase 重建复数频谱
→ IFFT
→ 零初始化残差缩放
```

频域分支输出：

```text
out = y + γ × FreMLP(y)
```

其中 `γ` 初始化为 0。

不采用当前 `FusedEnhanceBlock` 中的：

```text
y + γ × (y × FreMLP(y))
```

原因是逐元素乘法可能放大特征幅值，且不属于 DarkIR 的标准 FreMLP 残差形式。稳健版采用更直接的加性残差。

### 6.5 数值稳定性

频域计算需要特别处理混合精度：

- FFT 分支内部将输入临时转换为 `float32`。
- IFFT 后转换回原输入 dtype。
- 在幅度与相位计算中使用小常数避免异常数值。
- 单元测试必须验证输出不存在 `NaN` 或 `Inf`。

### 6.6 DarkIR 模块不负责的内容

- 不输出恢复后的 RGB 图像。
- 不使用 DarkIR 的 L1、LPIPS、edge loss 或低光照引导损失。
- 不改变 SAM2-UNet 的分割任务定义。
- 不将低光照增强与去模糊强制解释为当前分割模型的显式子任务。

在论文表述中，应称为“受 DarkIR 启发的空间—频域特征增强”，不能称为完整 DarkIR 复现。

---

## 7. FeatureBridge 设计

### 7.1 职责

`FeatureBridge` 是 Hiera 多尺度特征与 U-Net 解码器之间的唯一新增接口：

```python
class FeatureBridge(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=64,
        projection_mode="dynamic",
        enable_darkir=True,
        num_experts=4,
    ):
        ...

    def forward(self, x, return_router=False):
        ...
```

### 7.2 支持的模式

| `bridge_mode` | 投影模块 | DarkIR 增强 | 用途 |
|---|---|---|---|
| `rfb` | 原 `RFB_modified` | 否 | 原始 SAM2-UNet 基线 |
| `static` | 普通 `1×1 Conv` | 否 | 最小通道投影控制组 |
| `parameternet` | ParameterNet 动态投影 | 否 | 验证 ParameterNet 独立贡献 |
| `darkir` | 普通 `1×1 Conv` | 是 | 验证 DarkIR 独立贡献 |
| `full` | ParameterNet 动态投影 | 是 | 完整融合模型 |

该模式设计是可归因实验的关键。没有 `static` 控制组时，无法判断改进来自动态路由、DarkIR 增强，还是仅仅来自用更简单模块替换 RFB。

---

## 8. 完整模型设计

### 8.1 保留的 SAM2-UNet 结构

融合模型必须保留：

- `sam2_hiera_l.yaml` 对应的 Hiera-L 编码器；
- 删除 SAM2 原 mask decoder、prompt encoder 和 memory 相关模块的逻辑；
- 冻结原始 Hiera 参数；
- 插入中间维度为 32 的 Adapter；
- 原三级 U-Net 解码器；
- `out`、`out1`、`out2` 三个输出。

### 8.2 替换的结构

原模型中的：

```python
self.rfb1 = RFB_modified(144, 64)
self.rfb2 = RFB_modified(288, 64)
self.rfb3 = RFB_modified(576, 64)
self.rfb4 = RFB_modified(1152, 64)
```

在融合模型中替换为：

```python
self.bridge1 = FeatureBridge(144, 64, ...)
self.bridge2 = FeatureBridge(288, 64, ...)
self.bridge3 = FeatureBridge(576, 64, ...)
self.bridge4 = FeatureBridge(1152, 64, ...)
```

### 8.3 前向传播

```python
x1, x2, x3, x4 = self.encoder(x)

f1 = self.bridge1(x1)
f2 = self.bridge2(x2)
f3 = self.bridge3(x3)
f4 = self.bridge4(x4)

x = self.up1(f4, f3)
out1 = ...
x = self.up2(x, f2)
out2 = ...
x = self.up3(x, f1)
out = ...

return out, out1, out2
```

---

## 9. 参数量与计算量预期

### 9.1 ParameterNet 动态投影

四级普通 `1×1 Conv` 投影的卷积参数约为：

```text
(144 + 288 + 576 + 1152) × 64 = 138,240
```

使用四专家动态卷积后，专家卷积参数约为：

```text
138,240 × 4 = 552,960
```

路由器会增加少量参数和计算。主要空间卷积仍只执行一次，因此理论主卷积 FLOPs 接近普通 `1×1 Conv`。

### 9.2 与原 RFB 对比

现有四个 `RFB_modified` 合计约有 222 万参数。即使加入四专家动态投影和四个 DarkIR 增强块，融合桥接部分预计仍可控制在原 RFB 参数量附近或以下。

最终结论必须以实际统计为准，同时报告：

- 总参数量；
- 可训练参数量；
- MACs/FLOPs；
- 峰值显存；
- 单张和批量推理延迟；
- 吞吐量。

ParameterNet 的理论 FLOPs 优势不一定自动转化为更低 GPU 延迟，因此不能只报告 FLOPs。

---

## 10. 训练策略

### 10.1 参数训练范围

保持与 SAM2-UNet 一致：

- Hiera 原始参数：冻结；
- Adapter：可训练；
- FeatureBridge：可训练；
- U-Net 解码器与分割头：可训练。

### 10.2 损失函数

不引入 DarkIR 图像恢复损失。继续采用 SAM2-UNet 分割损失：

```text
Lseg = Lweighted-IoU + LBCE
Ltotal = Lseg(out) + Lseg(out1) + Lseg(out2)
```

若现有训练工程已经设置了不同的辅助损失权重，所有消融实验必须保持相同权重。

### 10.3 稳定性措施

- DarkIR 增强块的 `β` 和 `γ` 使用零初始化。
- 动态路由初始为均匀分配。
- 使用梯度裁剪，例如全局范数 `1.0`。
- 监控每层专家平均使用率和路由熵。
- 若路由长期完全坍缩，再单独实验轻量负载均衡损失；首轮实验不默认加入该损失，以免引入额外变量。

### 10.4 推荐训练顺序

1. 重跑 `rfb` 基线，确认数据和训练流程可靠。
2. 训练 `static`，建立简单投影控制组。
3. 分别训练 `parameternet` 和 `darkir`。
4. 训练 `full` 完整融合模型。
5. 完成主消融后，再调整专家数量和 DarkIR 分支。

所有模型使用相同的数据划分、增强方式、训练轮数、随机种子集合和评价代码。

---

## 11. 消融实验设计

### 11.1 核心消融矩阵

| 实验 | 投影 | 空间分支 | 频域分支 | 目的 |
|---|---|---|---|---|
| A | 原 RFB | 否 | 否 | SAM2-UNet 基线 |
| B | 静态 1×1 | 否 | 否 | 最小控制组 |
| C | ParameterNet 动态投影 | 否 | 否 | ParameterNet 独立贡献 |
| D | 静态 1×1 | 是 | 是 | DarkIR 独立贡献 |
| E | ParameterNet 动态投影 | 是 | 是 | 完整融合模型 |

### 11.2 DarkIR 分支消融

在完整融合模型基础上测试：

- 仅空间分支；
- 仅频域分支；
- 空间与频域共同使用；
- dilation `(1, 4, 9)` 与其他组合对比。

### 11.3 ParameterNet 消融

测试专家数量：

```text
M = 1、2、4、8
```

其中：

- `M=1` 用于验证动态多专家的必要性；
- `M=4` 为默认配置；
- `M=8` 仅用于观察容量增加是否仍有收益。

同时记录专家使用率，检查是否出现一个专家长期占据绝大多数路由权重。

### 11.4 对已有实验文件的比较

可将 `SAM2UNet_dblock_dat_fused_rfbhou.py` 作为额外对照组，回答：

- 真正的 ParameterNet 动态卷积是否优于 `DynamicAdaptiveTanh`？
- 稳健版加性 FreMLP 残差是否优于现有乘性频域融合？
- 将 DarkIR 空间和频域机制放在统一 FeatureBridge 中是否更易训练？

该对照不属于核心论文消融，避免影响主结论的清晰度。

---

## 12. 测试设计

### 12.1 ParameterNetDynamicProjection 单元测试

必须验证：

- 输入和输出尺寸正确；
- 路由权重形状为 `[B, M]`；
- 每个样本的路由权重之和接近 1；
- 所有专家和路由器均能收到梯度；
- `M=1` 且权重复制时，结果可与普通 `1×1 Conv` 对齐；
- batch size 为 1 和大于 1 时均可执行；
- 输出不存在 `NaN` 或 `Inf`。

### 12.2 DarkIRFeatureEnhancer 单元测试

必须验证：

- 输入输出尺寸一致；
- `β=0`、`γ=0` 时初始输出接近输入；
- 空间分支使用严格 depth-wise convolution，即 `groups=channels`；
- FFT/IFFT 后输出为实数张量；
- float32 和混合精度路径均无异常；
- 反向传播可执行。

### 12.3 完整模型测试

必须验证：

- 五种 `bridge_mode` 均可实例化；
- 默认返回接口始终为 `(out, out1, out2)`；
- 输入 `352×352` 时三个输出尺寸与基线一致；
- 加载基线 checkpoint 时能够使用 `strict=False` 初始化公共部分；
- 新模型 checkpoint 保存 `bridge_mode` 和 `num_experts` 配置；
- `SAM2UNet.py` 的基线行为未被修改。

---

## 13. 评价指标与验收标准

### 13.1 分割效果

根据实际任务至少报告：

- mIoU；
- Dice/F1；
- MAE；
- SAM2-UNet 原实验使用的任务相关指标。

至少使用三个固定随机种子，报告均值与标准差。

### 13.2 效率

报告：

- 总参数量和可训练参数量；
- MACs/FLOPs；
- 峰值显存；
- 单张推理延迟；
- 固定 batch size 下吞吐量。

### 13.3 完整融合模型采用门槛

满足以下条件后，才建议用完整融合模型替换基线：

1. `full` 在主要分割指标上稳定优于 `static`。
2. `full` 至少优于 `parameternet` 或 `darkir` 中的单模块模型，证明融合存在收益。
3. 目标是相对 `rfb` 基线提升至少 `0.5` 个百分点；若效果近似，则必须体现明确的参数量、显存或速度优势。
4. 推理延迟相对 `rfb` 基线的增加不超过约 15%，或有足够的精度收益解释额外开销。
5. 路由器没有明显专家坍缩，频域分支没有数值异常。

---

## 14. 风险与应对

| 风险 | 原因 | 应对 |
|---|---|---|
| 动态卷积实际延迟较高 | batch-grouped convolution 对硬件不一定友好 | 同时测量 FLOPs 和真实延迟；必要时提供静态部署版本 |
| 专家路由坍缩 | Softmax 长期只选择少数专家 | 监控使用率；必要时增加轻量平衡损失 |
| FFT 分支混合精度异常 | 半精度 FFT 和复数操作可能不稳定 | FFT 内部固定使用 float32 |
| 新模块导致训练初期震荡 | 同时替换 RFB 并加入增强模块 | 使用零初始化残差和均匀路由初始化 |
| 性能提升无法归因 | 多个模块一起修改 | 使用五种 bridge mode 和分支消融 |
| 误称完整复现 | 当前任务与原论文任务不同 | 明确使用“受 DarkIR/ParameterNet 启发并适配分割任务” |

---

## 15. 不采用的方案

### 15.1 直接修改 `SAM2UNet.py`

不采用。该文件需要作为论文基线，直接修改会失去可靠对照。

### 15.2 继续在 `SAM2UNet_dblock_dat_fused_rfbhou.py` 中叠加模块

不采用。该文件已经混合 DBlock_DAT、DynamicAdaptiveTanh 和 FusedEnhanceBlock，继续叠加真正的 ParameterNet 动态卷积会使实验难以归因。

### 15.3 对每个专家分别卷积再融合输出

不采用。该实现会将主卷积 FLOPs 放大到约 M 倍，不符合 ParameterNet 的核心设计。

### 15.4 修改 Hiera 内部 Attention 或 Adapter

稳健版不采用。该方案可能提高创新程度，但会破坏当前清晰的预训练迁移路径，并显著扩大实验范围。

### 15.5 加入 DarkIR 图像恢复辅助头和恢复损失

稳健版不采用。现有数据未必包含低光照或模糊图像的清晰配对标签，强行加入恢复任务可能导致训练目标错配。

---

## 16. 实施顺序

建议按照以下顺序实施：

1. 新建 `SAM2UNet_darkir_parameternet.py`，先实现静态 `FeatureBridge` 和模型模式切换。
2. 实现并测试 `ParameterNetDynamicProjection`。
3. 实现并测试 DarkIR 空间分支。
4. 实现并测试 DarkIR 频域分支。
5. 组合 `DarkIRFeatureEnhancer` 和完整 `FeatureBridge`。
6. 接入 SAM2-UNet 编码器与原 U-Net 解码器。
7. 完成五种核心模式的集成测试。
8. 运行效率基准与核心消融实验。
9. 仅在核心结果可靠后，开展专家数量和分支结构的次级消融。

---

## 17. 最终设计摘要

稳健版不直接改动两个现有 Python 文件，而是新增独立融合模型：

```text
冻结 SAM2 Hiera-L
    ↓
保留原 Adapter
    ↓
ParameterNet 多专家动态 1×1 投影
    ↓
DarkIR 风格空间 Di-SpAM + 频域 FreMLP
    ↓
保留原 U-Net 解码器与深监督输出
```

该方案真正实现了 ParameterNet 的动态卷积核心，而不是将普通动态激活误认为 ParameterNet；同时提取 DarkIR 可迁移到分割任务的空间与频域机制，但不引入任务不匹配的图像恢复损失。通过独立新文件、统一 FeatureBridge 和完整消融矩阵，可以在控制风险的同时验证两类创新是否对 SAM2-UNet 分割有效。
