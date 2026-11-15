"""可观测层：终端渲染器与格式化工具。

设计要点：
- 零依赖：曲线用 Unicode 块字符（``▁▂▃▄▅▆▇█``）画成"火花线"；
  终端编码不支持 Unicode（如 GBK 控制台）时自动退化为 ASCII 字符。
- 渲染器实现 Run 的全部展示回调，替换渲染后端（TensorBoard、web dashboard）
  时训练代码零改动。
"""
from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["spark", "fmt_num", "fmt_duration", "TerminalRenderer", "SPARK_BLOCKS_UNICODE", "SPARK_BLOCKS_ASCII"]

SPARK_BLOCKS_UNICODE = "▁▂▃▄▅▆▇█"
SPARK_BLOCKS_ASCII = "._-=+*#%@"


def _blocks_for(stream: Any) -> str:
    """根据流编码选择可安全输出的块字符集。"""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        SPARK_BLOCKS_UNICODE.encode(encoding)
        return SPARK_BLOCKS_UNICODE
    except (UnicodeEncodeError, LookupError):
        return SPARK_BLOCKS_ASCII


def spark(seq: Sequence[float], width: int = 48, stream: Any = None) -> str:
    """把数值序列渲染成火花线，只取最后 ``width`` 个点。

    序列中的 NaN/Inf 与非数值会被自动剔除，绝不让渲染层打断训练。
    """
    blocks = _blocks_for(stream if stream is not None else sys.stdout)
    values: List[float] = []
    for v in list(seq)[-width * 2:]:  # 多取一倍，过滤后仍够 width 个
        if isinstance(v, (int, float)) and math.isfinite(v):
            values.append(float(v))
    values = values[-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    n = len(blocks)
    if span < 1e-12:
        idx = [n // 2] * len(values)
    else:
        idx = [min(n - 1, int((v - lo) / span * n)) for v in values]
    return "".join(blocks[i] for i in idx)


def fmt_num(x: float) -> str:
    """紧凑数值格式：0.5231 / 1.2e+03 / nan。"""
    try:
        if x != x or x in (float("inf"), float("-inf")):
            return str(x)
    except TypeError:
        return str(x)
    if abs(x) >= 10000 or (0 < abs(x) < 1e-4):
        return "{:.3g}".format(x)
    s = "{:.4f}".format(x).rstrip("0").rstrip(".")
    return s if s else "0"


def fmt_duration(seconds: float) -> str:
    """秒数人性化：'8m32s' / '1h03m' / '45s'。"""
    s = int(seconds)
    if s < 60:
        return "{}s".format(s)
    if s < 3600:
        return "{}m{:02d}s".format(s // 60, s % 60)
    return "{}h{:02d}m".format(s // 3600, (s % 3600) // 60)


class TerminalRenderer:
    """默认渲染后端。

    - 有 TTY 时单行原地刷新（``\\r``），训练界面只有一行；
    - 无 TTY（重定向/CI）时降级为每 N 步打一行日志；
    - ``quiet=True`` 完全静默，只记录不打印。
    """

    def __init__(self, stream: Any = None, quiet: bool = False,
                 log_interval: int = 100, spark_width: int = 24) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.quiet = quiet
        self.log_interval = log_interval
        self.spark_width = spark_width
        self._tail: List[float] = []
        self._last_was_inline = False

    # -- 内部 ---------------------------------------------------------------

    def _write(self, text: str, inline: bool = False) -> None:
        if self.quiet:
            return
        try:
            if inline:
                self.stream.write("\r" + text)
                self.stream.flush()
            else:
                if self._last_was_inline:
                    self.stream.write("\n")
                self.stream.write(text + "\n")
                self.stream.flush()
            self._last_was_inline = inline
        except (UnicodeEncodeError, OSError):
            pass  # 编码不支持/流关闭时静默，绝不打断训练

    def _finish_inline(self) -> None:
        if self._last_was_inline and not self.quiet:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except OSError:
                pass
            self._last_was_inline = False

    # -- Run 回调 -----------------------------------------------------------

    def on_start(self, run_name: str, run_dir: str) -> None:
        self._write("[trainkeeper] 任务 '{}' 开始，存档目录：{}".format(run_name, run_dir))

    def on_resume(self, run_name: str, epoch: int, step_in_epoch: int) -> None:
        self._finish_inline()
        self._write("[trainkeeper] 检测到断点，任务 '{}' 从 epoch {} (step {}) 续跑。".format(
            run_name, epoch, step_in_epoch))

    def on_log(self, epoch: int, step: int, metrics: Dict[str, float],
               max_epochs: Optional[int]) -> None:
        if "loss" in metrics and isinstance(metrics["loss"], (int, float)):
            self._tail.append(float(metrics["loss"]))
        parts = ["e{}".format(epoch + 1)]
        if max_epochs:
            parts[0] = "e{}/{}".format(epoch + 1, max_epochs)
        parts.append("s{}".format(step))
        for key in ("loss", "acc"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                parts.append("{} {}".format(key, fmt_num(metrics[key])))
        extra_keys = [k for k in metrics if k not in ("loss", "acc", "lr")]
        if "lr" in metrics:
            parts.append("lr {}".format(fmt_num(metrics["lr"])))
        if len(extra_keys) > 0:
            parts.extend("{} {}".format(k, fmt_num(metrics[k])) for k in extra_keys[:3])
        line = " | ".join(parts)
        curve = spark(self._tail, width=self.spark_width, stream=self.stream)
        if curve:
            line = line + "  " + curve
        inline = hasattr(self.stream, "isatty") and self.stream.isatty()
        if not inline:
            if step % max(1, self.log_interval) != 0:
                return
        self._write(line, inline=inline)

    def on_diagnosis(self, diagnosis: Any) -> None:
        self._finish_inline()
        self._write(diagnosis.format())

    def on_epoch_end(self, epoch: int) -> None:
        pass  # 单行刷新模式下 epoch 边界无需换行，由 on_log 的内联光标体现

    def on_end(self, status: str, reason: Optional[str], summary: Dict[str, Any]) -> None:
        self._finish_inline()
        text = "[trainkeeper] 训练结束（{}）。".format(status)
        if reason:
            text += "原因：{}".format(reason)
        elapsed = summary.get("elapsed")
        if elapsed is not None:
            text += " 用时 {}".format(fmt_duration(elapsed))
        self._write(text)

    def on_report(self, report_text: str) -> None:
        self._finish_inline()
        self._write(report_text)
