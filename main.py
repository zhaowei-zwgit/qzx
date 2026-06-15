"""SAM2-UNet 息肉分割实验的根目录统一运行入口。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence


# 项目根目录与源码目录，用于保证从任意工作目录启动时都能正确定位文件。
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "polyp_train.json"

# 支持的特征桥接消融模式，与 sam2unet.experiment 中的模型配置保持一致。
BRIDGE_MODES = ("rfb", "static", "parameternet", "darkir", "full")


def _parser() -> argparse.ArgumentParser:
    """创建根入口命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="运行 SAM2-UNet 小规模测试、正式训练或模型评估。"
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "train", "evaluate"),
        default="smoke",
        help="运行模式，默认使用 smoke 小规模测试",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device", default="auto", help="运行设备：auto、cpu、cuda 或 cuda:N"
    )
    parser.add_argument("--bridge-mode", choices=BRIDGE_MODES, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="仅用于 train 模式")
    parser.add_argument("--resume", type=Path, default=None, help="仅用于 train 模式")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="evaluate 模式必须提供"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="禁用 TensorBoard 实时监控和 tqdm 进度条",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析参数，并检查各运行模式专属参数是否合法。"""
    parser = _parser()
    args = parser.parse_args(argv)

    # evaluate 必须指定待评估 checkpoint，其他模式不应误用该参数。
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--checkpoint is required when --mode evaluate")
    if args.mode != "evaluate" and args.checkpoint is not None:
        parser.error("--checkpoint can only be used when --mode evaluate")

    # epochs 和 resume 只控制正式训练流程。
    if args.mode != "train" and (args.epochs is not None or args.resume is not None):
        parser.error("--epochs and --resume can only be used when --mode train")
    return args


def _append_option(arguments: list[str], name: str, value: object | None) -> None:
    """仅在参数值不为空时，将选项转发给底层实验入口。"""
    if value is not None:
        arguments.extend((name, str(value)))


def build_experiment_argv(args: argparse.Namespace) -> list[str]:
    """将根入口参数转换为 sam2unet.experiment 使用的子命令格式。"""
    arguments = [
        args.mode,
        "--config",
        str(args.config),
        "--device",
        args.device,
    ]
    _append_option(arguments, "--bridge-mode", args.bridge_mode)
    _append_option(arguments, "--output-root", args.output_root)
    _append_option(arguments, "--limit-train", args.limit_train)
    _append_option(arguments, "--limit-test", args.limit_test)
    if args.mode == "train":
        _append_option(arguments, "--epochs", args.epochs)
        _append_option(arguments, "--resume", args.resume)
        if getattr(args, "no_monitor", False):
            arguments.append("--no-monitor")
    elif args.mode == "evaluate":
        _append_option(arguments, "--checkpoint", args.checkpoint)
    return arguments


def _load_experiment_main() -> Callable[[Sequence[str] | None], int]:
    """延迟导入实验入口，避免执行 --help 时提前加载 PyTorch 等依赖。"""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from sam2unet.experiment import main as experiment_main

    return experiment_main


def main(
    argv: Sequence[str] | None = None,
    experiment_main: Callable[[Sequence[str] | None], int] | None = None,
) -> int:
    """执行选定实验流程，并输出开始时间、结束时间和总耗时。"""
    args = parse_args(argv)
    forwarded = build_experiment_argv(args)
    run_experiment = experiment_main or _load_experiment_main()

    # 使用 datetime 输出易读时间，使用 perf_counter 精确计算实际耗时。
    started_at = datetime.now()
    started_clock = perf_counter()
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Start time: {started_at:%Y-%m-%d %H:%M:%S}")
    try:
        return run_experiment(forwarded)
    finally:
        ended_at = datetime.now()
        duration = perf_counter() - started_clock
        print(f"End time: {ended_at:%Y-%m-%d %H:%M:%S}")
        print(f"Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    raise SystemExit(main())
