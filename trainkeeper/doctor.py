"""诊断层：对训练指标序列做在线体检。

Doctor 维护一组可插拔的 :class:`DiagnosisRule`。每条规则独立观察指标流，
命中时产出 :class:`Diagnosis`（现象描述 + 大白话处置建议）。内置四条规则：

- :class:`NaNLossRule`      loss 出现 NaN/Inf
- :class:`DivergenceRule`   loss 持续上升（发散）
- :class:`SpikeRule`        loss 出现异常尖峰
- :class:`PlateauRule`      loss 长期停滞

自定义规则只需实现 ``observe(step, metrics) -> List[Diagnosis]`` 并调用
``doctor.add_rule(rule)``。
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "Diagnosis", "DiagnosisRule", "Doctor",
    "NaNLossRule", "DivergenceRule", "SpikeRule", "PlateauRule",
]

LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"


@dataclass(frozen=True)
class Diagnosis:
    """一次诊断结论。"""

    level: str          # "info" | "warn" | "error"
    kind: str           # 规则名，如 "divergence"
    message: str        # 现象描述
    advice: str         # 处置建议
    step: int           # 发现时的全局步数

    def format(self) -> str:
        tag = {"info": "[提示]", "warn": "[警告]", "error": "[严重]"}[self.level]
        return "{} {}\n       建议：{}".format(tag, self.message, self.advice)


@runtime_checkable
class DiagnosisRule(Protocol):
    """诊断规则契约。"""

    name: str

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:
        """消费一步指标，返回命中的诊断（可为空列表）。"""
        ...


def _slope(values: List[float]) -> float:
    """最小二乘拟合的每步斜率。"""
    n = len(values)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(values) / n
    num = sum((i - mx) * (v - my) for i, v in enumerate(values))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


class _RuleBase:
    """规则基类：指标缓存 + 冷却抑制（避免同一问题刷屏）。"""

    name = "base"

    def __init__(self, key: str, window: int, cooldown: int) -> None:
        self.key = key
        self.window = window
        self.cooldown = cooldown
        self._buffer: Deque[float] = deque(maxlen=window)
        self._last_fired: int = -(10 ** 9)

    def _should_fire(self, step: int) -> bool:
        if step - self._last_fired < self.cooldown:
            return False
        self._last_fired = step
        return True

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:  # pragma: no cover
        raise NotImplementedError


class NaNLossRule(_RuleBase):
    """监控指标是否出现 NaN / Inf。出现即升级为 error。"""

    def __init__(self, key: str = "loss", cooldown: int = 50) -> None:
        super().__init__(key=key, window=1, cooldown=cooldown)

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:
        value = metrics.get(self.key)
        if value is None:
            return []
        try:
            bad = not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value)
        except TypeError:
            bad = True
        if not bad:
            return []
        return [Diagnosis(
            level=LEVEL_ERROR,
            kind="nan_loss",
            message="{} 在 step {} 出现 NaN/Inf，训练已经 numerically 崩坏。".format(self.key, step),
            advice="最常见原因：学习率过大、梯度未裁剪、数据含异常值、或对空 batch 求了 log。"
                   "处置：学习率降一个数量级；加入梯度裁剪（如 clip_grad_norm_(max_norm=1.0)）；"
                   "检查数据管道是否产生 inf/nan 输入。",
            step=step,
        )]


class DivergenceRule(_RuleBase):
    """监控 loss 是否持续上升（相对斜率超过阈值）。"""

    def __init__(self, key: str = "loss", window: int = 50,
                 threshold: float = 0.005, cooldown: int = 200) -> None:
        super().__init__(key=key, window=window, cooldown=cooldown)
        self.threshold = threshold

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:
        value = metrics.get(self.key)
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return []
        self._buffer.append(float(value))
        if len(self._buffer) < self.window:
            return []
        recent = list(self._buffer)
        mean = sum(recent) / len(recent)
        if mean <= 0:
            return []
        rel_slope = _slope(recent) / mean
        if rel_slope <= self.threshold:
            return []
        if not self._should_fire(step):
            return []
        return [Diagnosis(
            level=LEVEL_WARN,
            kind="divergence",
            message="{} 近 {} 步相对斜率为 {:+.2%}/步，呈发散趋势。".format(
                self.key, self.window, rel_slope),
            advice="典型发散信号：降低学习率、开启梯度裁剪；"
                   "或用 run.replay_to() 回退到上一个稳定 epoch 重新来过。",
            step=step,
        )]


class SpikeRule(_RuleBase):
    """监控 loss 是否出现远超中位数的孤立尖峰。"""

    def __init__(self, key: str = "loss", window: int = 50,
                 factor: float = 6.0, cooldown: int = 200) -> None:
        super().__init__(key=key, window=window + 1, cooldown=cooldown)
        self.factor = factor

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:
        value = metrics.get(self.key)
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return []
        self._buffer.append(float(value))
        buf = list(self._buffer)
        if len(buf) < self.window:
            return []
        history = buf[:-1]
        median = statistics.median(history)
        if median <= 0 or buf[-1] < self.factor * median:
            return []
        if not self._should_fire(step):
            return []
        return [Diagnosis(
            level=LEVEL_WARN,
            kind="spike",
            message="{} 出现异常尖峰（{}，为中位数 {:.4g} 的 {:.1f} 倍）。".format(
                self.key, buf[-1], median, buf[-1] / median),
            advice="多为脏 batch、数值不稳定或学习率局部过大。若频繁出现，"
                   "检查数据增强管线与 batch 组成；也可考虑跳过异常 batch。",
            step=step,
        )]


class PlateauRule(_RuleBase):
    """监控 loss 是否长期停滞（波动幅度低于阈值）。"""

    def __init__(self, key: str = "loss", window: int = 150,
                 tolerance: float = 0.02, cooldown: int = 500) -> None:
        super().__init__(key=key, window=window, cooldown=cooldown)
        self.tolerance = tolerance

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:
        value = metrics.get(self.key)
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return []
        self._buffer.append(float(value))
        if len(self._buffer) < self.window:
            return []
        recent = list(self._buffer)
        mean = sum(recent) / len(recent)
        if mean <= 0:
            return []
        spread = (max(recent) - min(recent)) / mean
        if spread > self.tolerance:
            return []
        if not self._should_fire(step):
            return []
        return [Diagnosis(
            level=LEVEL_INFO,
            kind="plateau",
            message="{} 已连续 {} 步几乎不再变化（波动仅 {:.2%}）。".format(
                self.key, self.window, spread),
            advice="模型可能已收敛或陷入平坦区。建议：降低学习率 / 更换调度器 / "
                   "增强数据；若已达标，用 budget 或 until 让训练体面收工。",
            step=step,
        )]


class Doctor:
    """训练医生：聚合多条规则，一次 ``observe`` 返回全部命中诊断。

    Example:
        doctor = Doctor()
        doctor.add_rule(MyCustomRule())
        issues = doctor.observe(step=1234, metrics={"loss": 3.2})
    """

    def __init__(self, rules: Optional[List[DiagnosisRule]] = None) -> None:
        if rules is None:
            rules = [
                NaNLossRule(),
                DivergenceRule(),
                SpikeRule(),
                PlateauRule(),
            ]
        self._rules: List[DiagnosisRule] = list(rules)

    def add_rule(self, rule: DiagnosisRule) -> None:
        """注册自定义诊断规则。"""
        self._rules.append(rule)

    @property
    def rule_names(self) -> List[str]:
        return [getattr(r, "name", type(r).__name__) for r in self._rules]

    def observe(self, step: int, metrics: Dict[str, Any]) -> List[Diagnosis]:
        found: List[Diagnosis] = []
        for rule in self._rules:
            try:
                found.extend(rule.observe(step, metrics))
            except Exception:  # 规则出错绝不能打断训练
                continue
        return found
