"""可复现层：全局随机源的状态快照与恢复（"RNG 时光机"）。

深度学习实验不可复现的根源在于随机性散落在多个全局状态里：
``random``、``numpy.random``、``torch`` 的 CPU/CUDA 生成器，乃至用户自定义的
采样器。Ledger 把它们统一抽象为可插拔的 :class:`RandomSource`，在 epoch 边界
自动快照、按需恢复，从而实现：

- 崩溃续跑后，数据管道看到与崩溃前完全相同的随机序列；
- ``run.replay_to(epoch)`` 把"整个世界的随机状态"拨回任意 epoch 结束那一刻。
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Protocol, runtime_checkable

__all__ = ["RandomSource", "Ledger", "StdLibRandomSource", "NumpyRandomSource", "TorchRandomSource"]


@runtime_checkable
class RandomSource(Protocol):
    """单个随机源的存取契约。实现它即可把任何随机状态纳入 Ledger 管理。"""

    name: str

    def get_state(self) -> object:
        """返回可 pickle 的状态对象。"""
        ...

    def set_state(self, state: object) -> None:
        """恢复状态。"""
        ...


class StdLibRandomSource:
    """Python 标准库 ``random`` 模块。"""

    name = "random"

    def get_state(self) -> object:
        return random.getstate()

    def set_state(self, state: object) -> None:
        random.setstate(state)  # type: ignore[arg-type]


class NumpyRandomSource:
    """``numpy.random`` 全局生成器（未安装 numpy 时自动跳过）。"""

    name = "numpy"

    def __init__(self) -> None:
        import numpy  # noqa: F401  延迟导入，保持库本身零依赖

    def get_state(self) -> object:
        import numpy
        return numpy.random.get_state()

    def set_state(self, state: object) -> None:
        import numpy
        numpy.random.set_state(state)


class TorchRandomSource:
    """``torch`` 的 CPU 全局生成器（未安装 torch 时自动跳过）。

    CUDA 端随机状态请通过 :meth:`Ledger.register` 注册自定义 source 补齐。
    """

    name = "torch"

    def __init__(self) -> None:
        import torch  # noqa: F401

    def get_state(self) -> object:
        import torch
        return torch.get_rng_state()

    def set_state(self, state: object) -> None:
        import torch
        torch.set_rng_state(state)


def _default_sources() -> List[RandomSource]:
    sources: List[RandomSource] = [StdLibRandomSource()]
    for cls in (NumpyRandomSource, TorchRandomSource):
        try:
            sources.append(cls())
        except ImportError:
            pass
    return sources


class Ledger:
    """随机源注册表 + 快照/恢复引擎。

    Example:
        ledger = Ledger()
        snap = ledger.snapshot()          # 拍下当前所有随机源
        ...  # 任何消耗随机数的操作
        ledger.restore(snap)              # 世界回到快照那一刻
    """

    def __init__(self, include_defaults: bool = True) -> None:
        self._sources: Dict[str, RandomSource] = {}
        if include_defaults:
            for src in _default_sources():
                self._sources[src.name] = src

    # -- 注册管理 -----------------------------------------------------------

    def register(self, source: RandomSource) -> None:
        """注册随机源；同名覆盖。"""
        self._sources[source.name] = source

    def unregister(self, name: str) -> None:
        self._sources.pop(name, None)

    def source_names(self) -> List[str]:
        return sorted(self._sources)

    # -- 快照 / 恢复 ---------------------------------------------------------

    def snapshot(self) -> Dict[str, object]:
        """拍下全部已注册随机源的状态。"""
        return {name: src.get_state() for name, src in self._sources.items()}

    def restore(self, snap: Dict[str, object]) -> None:
        """恢复快照。快照中不存在于当前注册表的源会被跳过。"""
        for name, state in snap.items():
            src = self._sources.get(name)
            if src is not None:
                src.set_state(state)

    def restore_partial(self, snap: Dict[str, object], names: Optional[List[str]] = None) -> None:
        """只恢复指定名称的随机源。"""
        targets = names if names is not None else list(snap)
        for name in targets:
            if name in snap and name in self._sources:
                self._sources[name].set_state(snap[name])
