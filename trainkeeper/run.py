"""核心层：Run —— 训练任务的完整生命周期管理。

Run 是 trainkeeper 的门面，把五个子系统编排成一个上下文管理器：

1. **断点续跑**（崩溃安全）：状态经 ``StateStore`` 原子落盘；进程崩了、被
   Ctrl-C 了，重跑同一脚本自动从断点继续（epoch 级权重工件 + step 级进度）。
2. **RNG 时光机**（可复现）：每个 epoch 结束由 ``Ledger`` 快照全部随机源；
   ``replay_to(epoch)`` 把世界拨回那一刻。
3. **预算控制**：``budget="10m"`` / ``until="loss<0.05"`` / ``run.stop()``。
4. **在线诊断**：``log()`` 即体检，``Doctor`` 规则可插拔。
5. **可观测**：终端火花线 + JSONL 事件流 + ``report()`` 报告 + ``tk`` CLI。

语义约定（务必阅读）：
- ``run.epoch`` 表示**已完成**的 epoch 数，从 0 开始；
- ``max_epochs`` 指总目标数（含续跑前已完成的）；
- resume 时权重恢复由用户负责：每个 epoch 调 ``run.keep(model)``，
  重启后用 ``run.kept()`` 取回最近一次保存的工件；
- ``until`` 依据**最近一次 log 的指标值**在 epoch 边界判定。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from operator import eq, ge, gt, le, lt
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .doctor import Diagnosis, Doctor
from .ledger import Ledger
from .spark import TerminalRenderer, fmt_duration, fmt_num, spark
from .store import HistoryLog, JsonStateStore, PickleArtifactStore

__all__ = [
    "Run", "RunConfig", "RunStatus", "EmergencyStop",
    "parse_budget", "parse_until", "load_summary",
]


# ---------------------------------------------------------------------------
# 状态机与配置
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    """Run 生命周期状态。"""

    CREATED = "created"          # 已构造，尚未进入 with 块
    RUNNING = "running"          # 训练进行中
    COMPLETED = "completed"      # 达到 max_epochs 或 until 条件
    STOPPED = "stopped"          # 预算用尽或 run.stop() 被调用
    INTERRUPTED = "interrupted"  # 被 KeyboardInterrupt 打断（进度已存档）
    ABORTED = "aborted"          # run.abort() 或其他异常终止


class EmergencyStop(Exception):
    """由 :meth:`Run.abort` 抛出，表示训练被管家判定必须终止。"""


@dataclass
class RunConfig:
    """Run 的全部可调参数。"""

    name: str                       # 任务名（即存档目录名）
    root: str = ".tkruns"           # 存档根目录
    resume: bool = True             # 存在断点时自动续跑
    quiet: bool = False             # 静默模式：只记录不渲染
    doctor: bool = True             # 是否启用在线诊断
    renderer: bool = True           # 是否启用终端渲染
    abort_on_nan: bool = False      # loss 出现 NaN 时立即 abort
    state_save_interval: int = 1    # 每 N 步落盘一次状态（越大 I/O 越少，断点粒度越粗）
    catch_interrupt: bool = True    # Ctrl-C 时存档并优雅退出


# ---------------------------------------------------------------------------
# 解析工具
# ---------------------------------------------------------------------------

_BUDGET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh])\s*$", re.IGNORECASE)
_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}
_UNTIL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_OPS = {"<": lt, "<=": le, ">": gt, ">=": ge, "==": eq}
_MIN_HINTS = ("loss", "err")  # 键名含这些字样的指标默认取最小值为最优


def parse_budget(text: str) -> float:
    """解析预算字符串（如 ``'45s' / '10m' / '2h'``）为秒数。"""
    m = _BUDGET_RE.match(text)
    if not m:
        raise ValueError("无法解析预算 {!r}，合法示例：'45s' / '10m' / '2h'".format(text))
    return float(m.group(1)) * _UNITS[m.group(2).lower()]


def parse_until(text: str) -> Tuple[str, str, float]:
    """解析停止条件（如 ``'loss<0.05'``）为 (键, 操作符, 阈值)。"""
    m = _UNTIL_RE.match(text)
    if not m:
        raise ValueError("无法解析停止条件 {!r}，合法示例：'loss<0.05' / 'val_acc>=0.9'".format(text))
    return m.group(1), m.group(2), float(m.group(3))


def _default_mode(key: str) -> str:
    low = key.lower()
    return "min" if any(h in low for h in _MIN_HINTS) else "max"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

HookFn = Callable[..., None]


class Run:
    """训练任务管家。

    Example::

        import trainkeeper as tk

        with tk.run("exp1") as run:
            for epoch in run.epochs(max_epochs=50, budget="10m"):
                for batch in run.steps(loader):
                    loss = train_one(batch)
                    run.log(loss=loss)
                run.keep(model.state_dict())
            run.report()
    """

    def __init__(
        self,
        name: str,
        *,
        root: str = ".tkruns",
        resume: bool = True,
        quiet: bool = False,
        doctor: bool = True,
        renderer: bool = True,
        abort_on_nan: bool = False,
        state_save_interval: int = 1,
        catch_interrupt: bool = True,
        state_store: Optional[Any] = None,
        artifact_store: Optional[Any] = None,
        history_log: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.dir = os.path.join(root, name)
        self._cfg = RunConfig(
            name=name, root=root, resume=resume, quiet=quiet, doctor=doctor,
            renderer=renderer, abort_on_nan=abort_on_nan,
            state_save_interval=max(1, int(state_save_interval)),
            catch_interrupt=catch_interrupt,
        )
        # 持久层（默认本地实现，可通过参数注入自定义后端）
        self.store = state_store or JsonStateStore(self.dir)
        self.artifacts = artifact_store or PickleArtifactStore(os.path.join(self.dir, "artifacts"))
        self.history = history_log or HistoryLog(os.path.join(self.dir, "history.jsonl"))
        # 子系统
        self.ledger = Ledger()
        self.doctor = Doctor() if doctor else None
        self.renderer = TerminalRenderer(quiet=quiet) if renderer else None
        # 断点恢复
        prior = self.store.read("state") if resume else None
        self.resumed = bool(prior)
        prior = prior or {}
        self.epoch: int = int(prior.get("epoch", 0))
        self.step_in_epoch: int = int(prior.get("step_in_epoch", 0))
        self.global_step: int = int(prior.get("global_step", 0))
        self.best: Dict[str, float] = dict(prior.get("best", {}))
        self.best_epoch: Dict[str, int] = dict(prior.get("best_epoch", {}))
        self.last_metrics: Dict[str, float] = dict(prior.get("last_metrics", {}))
        self.elapsed_before: float = float(prior.get("elapsed", 0.0))
        # 运行时
        self.status: RunStatus = RunStatus.CREATED
        self.stop_reason: Optional[str] = None
        self._hooks: Dict[str, List[HookFn]] = {}
        self._metric_modes: Dict[str, str] = {}
        self._stop_requested = False
        self._steps_since_save = 0
        self._t0 = time.time()

    # -- 工厂 ---------------------------------------------------------------

    @classmethod
    def attach(cls, run_dir: str) -> "Run":
        """以只读方式挂载一个已有 Run 目录（用于事后查看报告）。"""
        run_dir = os.path.normpath(run_dir)
        name = os.path.basename(run_dir)
        parent = os.path.dirname(run_dir) or "."
        return cls(name, root=parent, resume=False, quiet=True,
                   doctor=False, renderer=False)

    # -- 事件钩子 -----------------------------------------------------------

    def on(self, event: str, fn: HookFn) -> None:
        """注册事件回调。事件：start / resume / step / epoch_end / diagnosis / end。

        回调签名不固定，trainkeeper 以关键字参数尽力传入上下文，
        回调建议声明 ``**kwargs`` 以兼容后续版本。
        """
        self._hooks.setdefault(event, []).append(fn)

    def _emit(self, event: str, **kwargs: Any) -> None:
        for fn in self._hooks.get(event, []):
            try:
                fn(**kwargs)
            except Exception:
                continue  # 钩子故障不影响训练本体

    # -- 上下文管理器 ---------------------------------------------------------

    def __enter__(self) -> "Run":
        self.status = RunStatus.RUNNING
        if self.resumed:
            snap = self.artifacts.load("rng/latest")
            if snap is not None:
                self.ledger.restore(snap)
            if self.renderer is not None:
                self.renderer.on_resume(self.name, self.epoch, self.step_in_epoch)
            self._emit("resume", epoch=self.epoch, step_in_epoch=self.step_in_epoch)
        else:
            if self.renderer is not None:
                self.renderer.on_start(self.name, self.dir)
            self._emit("start", name=self.name, dir=self.dir)
        self._save_state()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            if self.status == RunStatus.RUNNING:
                self.status = RunStatus.COMPLETED
        elif exc_type is KeyboardInterrupt and self._cfg.catch_interrupt:
            self.status = RunStatus.INTERRUPTED
            self.stop_reason = "KeyboardInterrupt"
            if self.renderer is not None:
                self.renderer.on_end(
                    "interrupted",
                    "Ctrl-C 已打断，进度已存档；重跑同一脚本即可从 epoch {} step {} 续跑".format(
                        self.epoch, self.step_in_epoch),
                    self._summary(),
                )
        elif exc_type is not None and self.status != RunStatus.ABORTED:
            self.status = RunStatus.ABORTED
            self.stop_reason = "{}: {}".format(exc_type.__name__, exc)
        self._save_state()
        self._emit("end", status=self.status.value, reason=self.stop_reason)
        if (exc_type is not None
                and exc_type is KeyboardInterrupt
                and self._cfg.catch_interrupt):
            return True  # 优雅吞掉 Ctrl-C
        return False

    # -- 主循环 ---------------------------------------------------------------

    def epochs(
        self,
        max_epochs: Optional[int] = None,
        budget: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Iterator[int]:
        """epoch 主循环。三种停止条件至少给一个：

        - ``max_epochs``：总 epoch 数目标（含续跑前已完成的）；
        - ``budget``：时间预算，如 ``"10m"``；
        - ``until``：指标条件，如 ``"loss<0.05"``，在 epoch 边界依据最近一次
          log 的值判定。

        Yields:
            当前 epoch 序号（已完成数，从 0 开始）。
        """
        if max_epochs is None and budget is None and until is None:
            raise ValueError("epochs() 需要至少一个停止条件：max_epochs / budget / until")
        deadline: Optional[float] = None
        if budget is not None:
            deadline = time.time() + parse_budget(budget) - self.elapsed_before
        cond = parse_until(until) if until is not None else None
        stop_kind: Optional[str] = None  # "max_epochs" | "budget" | "until" | "stop"

        while not self._stop_requested and self.status != RunStatus.ABORTED:
            if max_epochs is not None and self.epoch >= max_epochs:
                stop_kind, self.stop_reason = "max_epochs", "达到 max_epochs={}".format(max_epochs)
                break
            if deadline is not None and time.time() >= deadline:
                stop_kind, self.stop_reason = "budget", "时间预算 {} 用尽".format(budget)
                break
            if cond is not None and self._until_met(*cond):
                key, op, value = cond
                stop_kind, self.stop_reason = "until", "满足条件 {} {} {}".format(key, op, value)
                break
            yield self.epoch
            self._end_of_epoch()

        if self.status == RunStatus.RUNNING:
            if stop_kind in ("max_epochs", "until"):
                self.status = RunStatus.COMPLETED
            else:
                self.status = RunStatus.STOPPED
                if self.stop_reason is None:
                    self.stop_reason = "run.stop()"

    def steps(self, iterable: Iterable[Any], resume: str = "skip") -> Iterator[Any]:
        """step 子循环，自动处理断点续跑时的步级跳过。

        Args:
            iterable: 任意可迭代对象（DataLoader、range、生成器均可）。
            resume: 续跑策略。``"skip"``（默认）跳过断点前已消费的项；
                ``"redo"`` 从头重跑当前 epoch。
        """
        it = iter(iterable)
        skip = self.step_in_epoch if (self.resumed and resume == "skip") else 0
        index = 0
        for item in it:
            if index < skip:
                index += 1
                continue
            self._emit("step", index=index, epoch=self.epoch)
            yield item
            index += 1
            self.step_in_epoch = index
            self._steps_since_save += 1
            if self._steps_since_save >= self._cfg.state_save_interval:
                self._save_state()
                self._steps_since_save = 0

    # -- 指标记录与诊断 -------------------------------------------------------

    def set_metric_mode(self, key: str, mode: str) -> None:
        """显式指定某指标的优化方向（``"min"`` 或 ``"max"``）。

        默认启发式：键名含 loss/err 取最小值最优，其余取最大值最优。
        """
        if mode not in ("min", "max"):
            raise ValueError("mode 必须是 'min' 或 'max'")
        self._metric_modes[key] = mode

    def log(self, **metrics: Any) -> List[Diagnosis]:
        """记录一步指标（核心心跳）。同时触发诊断、渲染与状态落盘。

        Returns:
            本步命中的诊断列表（供程序化使用）。
        """
        self.global_step += 1
        record: Dict[str, Any] = {
            "t": round(self._elapsed(), 3),
            "epoch": self.epoch,
            "step": self.global_step,
        }
        record.update(metrics)
        self.history.append(record)

        numeric: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key] = float(value)
        for key, value in numeric.items():
            self.last_metrics[key] = value
            mode = self._metric_modes.get(key) or _default_mode(key)
            current = self.best.get(key)
            improved = (
                current is None
                or (mode == "min" and value < current)
                or (mode == "max" and value > current)
            )
            if improved:
                self.best[key] = value
                self.best_epoch[key] = self.epoch

        diagnoses: List[Diagnosis] = []
        if self.doctor is not None:
            diagnoses = self.doctor.observe(self.global_step, metrics)
            for d in diagnoses:
                self.history.append({
                    "t": round(self._elapsed(), 3), "event": d.kind,
                    "level": d.level, "msg": d.message, "step": self.global_step,
                })
                if self.renderer is not None:
                    self.renderer.on_diagnosis(d)
                self._emit("diagnosis", diagnosis=d)
            if self._cfg.abort_on_nan and any(d.kind == "nan_loss" for d in diagnoses):
                self.abort("loss 出现 NaN（abort_on_nan=True）")

        if self.renderer is not None:
            max_ep = getattr(self, "_max_epochs_hint", None)
            self.renderer.on_log(self.epoch, self.global_step, metrics, max_ep)
        return diagnoses

    # -- 工件与时光机 ---------------------------------------------------------

    def keep(self, obj: Any, tag: Optional[str] = None) -> str:
        """保存任意可 pickle 工件（模型权重、优化器状态等）。

        每个 epoch 调一次即可获得完整的权重档案；
        续跑时用 :meth:`kept` 取回最近一份恢复权重。
        """
        epoch_label = self.epoch + 1
        key = "epoch{:04d}{}".format(epoch_label, "_" + tag if tag else "")
        path = self.artifacts.save(key, obj)
        self.artifacts.save("latest", obj)
        return path

    def kept(self) -> Optional[Any]:
        """取回最近一次 :meth:`keep` 保存的工件（无则返回 None）。"""
        return self.artifacts.load("latest")

    def replay_to(self, epoch: int) -> None:
        """RNG 时光机：把全部随机源拨回 ``epoch``（1-based）结束那一刻。

        之后消耗随机数的代码（数据打乱、增强、探索）将与该 epoch 之后的
        训练轨迹完全一致——实验复现的关键开关。
        """
        key = "epoch{:04d}".format(epoch)
        snap = self.artifacts.load("rng/" + key)
        if snap is None:
            snap = self.artifacts.load(key)  # 兼容旧命名
        if snap is None:
            raise FileNotFoundError(
                "没有 epoch {} 的随机状态档案。可用 epoch：{}".format(
                    epoch, ", ".join(self.rng_archive()) or "（无）"))
        self.ledger.restore(snap)

    def rng_archive(self) -> List[str]:
        """列出所有已存档的 RNG 快照 key（形如 ``epoch0003``）。"""
        out = []
        for key in self.artifacts.keys():
            base = key.split("/")[-1]
            if base.startswith("epoch"):
                out.append(base)
        return out

    # -- 控制流 ---------------------------------------------------------------

    def stop(self, reason: str = "run.stop()") -> None:
        """请求优雅停止：当前 epoch 跑完后不再开新 epoch。"""
        self._stop_requested = True
        self.stop_reason = reason

    def abort(self, reason: str) -> None:
        """立即终止训练（抛出 :class:`EmergencyStop`）。"""
        self.status = RunStatus.ABORTED
        self.stop_reason = reason
        raise EmergencyStop(reason)

    def budget_left(self, budget: str) -> float:
        """给定总预算，返回剩余秒数（可为负）。"""
        return parse_budget(budget) - self._elapsed()

    # -- 报告 -----------------------------------------------------------------

    def report(self, print_report: bool = True) -> Dict[str, Any]:
        """生成训练事后报告（打印并返回结构化数据）。"""
        summary = load_summary(self.dir)
        text = format_summary(summary)
        if print_report and self.renderer is not None:
            self.renderer.on_report(text)
        return summary

    # -- 内部 -----------------------------------------------------------------

    def _elapsed(self) -> float:
        return self.elapsed_before + (time.time() - self._t0)

    def _until_met(self, key: str, op: str, value: float) -> bool:
        if key not in self.last_metrics:
            return False
        return bool(_OPS[op](self.last_metrics[key], value))

    def _end_of_epoch(self) -> None:
        self.epoch += 1
        self.step_in_epoch = 0
        snap = self.ledger.snapshot()
        self.artifacts.save("rng/latest", snap)
        self.artifacts.save("rng/epoch{:04d}".format(self.epoch), snap)
        self._save_state()
        if self.renderer is not None:
            self.renderer.on_epoch_end(self.epoch)
        self._emit("epoch_end", epoch=self.epoch, metrics=dict(self.last_metrics))

    def _save_state(self) -> None:
        self.store.write("state", {
            "name": self.name,
            "status": self.status.value if self.status != RunStatus.RUNNING else "running",
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
            "elapsed": round(self._elapsed(), 3),
            "best": self.best,
            "best_epoch": self.best_epoch,
            "last_metrics": self.last_metrics,
            "updated_at": time.time(),
        })

    def _summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "elapsed": self._elapsed(),
        }

    # -- 兼容 with tk.run(...) ------------------------------------------------

    def __repr__(self) -> str:
        return "Run(name={!r}, epoch={}, global_step={}, status={})".format(
            self.name, self.epoch, self.global_step, self.status.value)


# ---------------------------------------------------------------------------
# 报告聚合（Run.report 与 CLI 共用）
# ---------------------------------------------------------------------------

def load_summary(run_dir: str) -> Dict[str, Any]:
    """聚合一个 Run 目录的全部信息为结构化报告。"""
    history_path = os.path.join(run_dir, "history.jsonl")
    recs = HistoryLog(history_path).read_all()
    metrics: List[float] = [r["loss"] for r in recs
                            if "loss" in r and isinstance(r["loss"], (int, float))]
    events = [r for r in recs if "event" in r]
    state = JsonStateStore(run_dir).read("state") or {}
    epoch_losses: Dict[int, List[float]] = {}
    for r in recs:
        if "loss" in r and isinstance(r["loss"], (int, float)):
            epoch_losses.setdefault(int(r.get("epoch", 0)), []).append(float(r["loss"]))
    means = [sum(v) / len(v) for _, v in sorted(epoch_losses.items())]
    return {
        "dir": run_dir,
        "state": state,
        "epochs_done": len(epoch_losses),
        "global_step": int(state.get("global_step", 0)) or len(recs),
        "elapsed": float(state.get("elapsed", 0.0)),
        "best": state.get("best", {}),
        "best_epoch": state.get("best_epoch", {}),
        "epoch_mean_loss": means,
        "num_records": len(recs),
        "num_events": len(events),
        "events": events,
    }


def format_summary(summary: Dict[str, Any]) -> str:
    """把 :func:`load_summary` 的结果渲染成终端报告文本。"""
    best = summary.get("best", {})
    best_epoch = summary.get("best_epoch", {})
    lines: List[str] = []
    lines.append("╭─ trainkeeper 报告 · {} ─╮".format(os.path.basename(summary["dir"])))
    lines.append("│ epochs {} · steps {} · 用时 {}".format(
        summary["epochs_done"], summary["global_step"],
        fmt_duration(summary["elapsed"])))
    if best:
        parts = []
        for key, value in best.items():
            parts.append("{} {} (e{})".format(key, fmt_num(value), best_epoch.get(key, "?")))
        lines.append("│ 最优指标：{}".format("，".join(parts)))
    curve = spark(summary.get("epoch_mean_loss") or [], width=40)
    if curve:
        lines.append("│ 每epoch平均loss：{}".format(curve))
    lines.append("│ 诊断事件：{} 条".format(summary["num_events"]))
    for ev in summary["events"][-5:]:
        lines.append("│   · [{}] {} {}".format(ev.get("level", "?"),
                                              ev.get("step", "?"), ev.get("msg", "")))
    lines.append("╰─ 目录：{} ─╯".format(summary["dir"]))
    return "\n".join(lines)
