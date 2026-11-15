"""trainkeeper —— 给 Python AI 训练加上"管家"的基础设施层。

五个子系统的对外接口::

    import trainkeeper as tk

    with tk.run("exp1") as run:              # 断点续跑 + 状态机
        for epoch in run.epochs(max_epochs=50, budget="10m", until="loss<0.05"):
            for batch in run.steps(loader):  # 步级断点
                loss = train_one(batch)
                run.log(loss=loss)           # 心跳 + 在线诊断
                run.keep(model.state_dict()) # 权重档案
            run.replay_to(23)                # RNG 时光机（按需）
        run.report()                         # 事后报告

可插拔协议：``StateStore`` / ``ArtifactStore`` / ``RandomSource`` /
``DiagnosisRule``，实现同名接口即可替换后端，训练代码零改动。
"""
from __future__ import annotations

from .doctor import (
    Diagnosis,
    DiagnosisRule,
    Doctor,
    DivergenceRule,
    NaNLossRule,
    PlateauRule,
    SpikeRule,
)
from .ledger import (
    Ledger,
    NumpyRandomSource,
    RandomSource,
    StdLibRandomSource,
    TorchRandomSource,
)
from .run import (
    EmergencyStop,
    Run,
    RunConfig,
    RunStatus,
    load_summary,
    parse_budget,
    parse_until,
)
from .spark import fmt_duration, fmt_num, spark
from .store import ArtifactStore, JsonStateStore, PickleArtifactStore, StateStore

__version__ = "0.1.0"
__all__ = [
    # 核心
    "run", "Run", "RunConfig", "RunStatus", "EmergencyStop",
    "parse_budget", "parse_until", "load_summary",
    # 可复现
    "Ledger", "RandomSource",
    "StdLibRandomSource", "NumpyRandomSource", "TorchRandomSource",
    # 诊断
    "Doctor", "Diagnosis", "DiagnosisRule",
    "NaNLossRule", "DivergenceRule", "SpikeRule", "PlateauRule",
    # 可观测
    "spark", "fmt_num", "fmt_duration",
    # 持久层
    "StateStore", "ArtifactStore", "JsonStateStore", "PickleArtifactStore",
    "__version__",
]


def run(name: str, **kwargs: object) -> Run:
    """构造一个 :class:`Run`。参数与 :class:`RunConfig` 一致。

    Example::

        with tk.run("exp1") as run:
            ...
    """
    return Run(name, **kwargs)  # type: ignore[arg-type]
