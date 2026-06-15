"""训练过程实时监控模块，集成 TensorBoard 日志与 tqdm 进度条。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

from tqdm import tqdm


class TrainingMonitor:
    """训练过程实时监控。

    提供两种实时反馈渠道：
    - tqdm 进度条：终端内实时显示 batch loss 和学习率
    - TensorBoard：Web 界面查看 loss 曲线、指标趋势、学习率调度等

    参数均为可选，可通过 enabled=False 完全禁用（空操作模式）。
    """

    def __init__(self, output_dir: Path | str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._writer = None
        self._pbar: tqdm | None = None

        if not enabled:
            return

        self._tb_dir = Path(output_dir) / "tb_logs"
        self._tb_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self._tb_dir))
        except ImportError:
            # tensorboard 未安装时降级为仅 tqdm 模式
            self._writer = None

    # ------------------------------------------------------------------
    # epoch 生命周期
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: int, total_batches: int) -> None:
        """每个 epoch 开始时调用，初始化进度条。"""
        if not self.enabled:
            return
        self._pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch}",
            leave=True,
            dynamic_ncols=True,
        )

    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        lr: float,
    ) -> None:
        """每个 batch 结束后调用，更新进度条并写入 TensorBoard。"""
        if not self.enabled:
            return

        # 更新 tqdm 进度条
        if self._pbar is not None:
            self._pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")
            self._pbar.update(1)

        # TensorBoard: batch 级别指标
        if self._writer is not None:
            global_step = (epoch - 1) * 10000 + batch_idx  # 全局步数近似
            self._writer.add_scalar("train/batch_loss", loss, global_step)
            self._writer.add_scalar("train/learning_rate", lr, global_step)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        evaluations: Mapping[str, Mapping[str, float]],
        average_dice: float,
    ) -> None:
        """每个 epoch 结束后调用，记录评估指标并关闭进度条。"""
        if not self.enabled:
            return

        # 关闭进度条
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        if self._writer is None:
            return

        # TensorBoard: epoch 级别训练指标
        self._writer.add_scalar("train/epoch_loss", train_loss, epoch)

        # TensorBoard: 各测试集评估指标
        for set_name, metrics in evaluations.items():
            for metric_name, value in metrics.items():
                tag = f"val/{set_name}/{metric_name}"
                self._writer.add_scalar(tag, value, epoch)

        # TensorBoard: 平均 Dice（用于快速对比）
        self._writer.add_scalar("val/average_dice", average_dice, epoch)

        # 打印 epoch 摘要
        eval_strs = []
        for set_name, metrics in evaluations.items():
            dice = metrics.get("dice", 0.0)
            iou = metrics.get("iou", 0.0)
            eval_strs.append(f"{set_name}: Dice={dice:.4f} IoU={iou:.4f}")
        summary = " | ".join(eval_strs)
        tqdm.write(
            f"Epoch {epoch} summary — loss={train_loss:.4f} | {summary}"
        )

    def on_train_end(self) -> None:
        """训练全部结束后调用，释放资源。"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------
    # 可选：可视化预测结果
    # ------------------------------------------------------------------

    def add_image(
        self, tag: str, image_tensor, epoch: int, dataformats: str = "CHW"
    ) -> None:
        """将图像（如预测结果对比）写入 TensorBoard。"""
        if self._writer is not None:
            self._writer.add_image(tag, image_tensor, epoch, dataformats=dataformats)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def tb_log_dir(self) -> Path | None:
        """TensorBoard 日志目录路径，未启用时返回 None。"""
        return self._tb_dir if self.enabled and self._writer is not None else None


__all__ = ["TrainingMonitor"]
