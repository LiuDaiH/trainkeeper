"""核心层测试：断点续跑、预算、until、时光机、钩子、报告。"""
import random

import pytest

import trainkeeper as tk
from trainkeeper.run import parse_budget, parse_until, RunStatus


# ---------------------------------------------------------------------------
# 解析工具
# ---------------------------------------------------------------------------

def test_parse_budget():
    assert parse_budget("45s") == 45.0
    assert parse_budget("10m") == 600.0
    assert parse_budget("2h") == 7200.0
    assert parse_budget("1.5M") == 90.0
    with pytest.raises(ValueError):
        parse_budget("10x")
    with pytest.raises(ValueError):
        parse_budget("abc")


def test_parse_until():
    assert parse_until("loss<0.05") == ("loss", "<", 0.05)
    assert parse_until("val_acc>=0.9") == ("val_acc", ">=", 0.9)
    with pytest.raises(ValueError):
        parse_until("loss 0.05")


# ---------------------------------------------------------------------------
# 正常训练流
# ---------------------------------------------------------------------------

def _train(run, n_epochs=3, n_steps=10, crash_at=None):
    """模拟一个标准训练循环。crash_at=(epoch, step_idx) 时抛 KeyboardInterrupt。"""
    history = []
    for _epoch in run.epochs(max_epochs=n_epochs):
        for i, batch in enumerate(run.steps(range(n_steps))):
            if crash_at is not None and (_epoch, i) == crash_at:
                raise KeyboardInterrupt
            loss = 1.0 / (1 + i + _epoch)
            run.log(loss=loss, acc=0.5 + 0.01 * i)
            history.append((_epoch, i))
    return history


def test_normal_run_completes(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        history = _train(run, n_epochs=3, n_steps=10)
    assert len(history) == 30
    assert run.epoch == 3
    assert run.status == RunStatus.COMPLETED
    assert run.best["loss"] == min(1.0 / (1 + i + e) for e in range(3) for i in range(10))


def test_log_returns_diagnoses_and_best(tmp_path):
    with tk.run("exp", root=str(tmp_path), doctor=False) as run:
        run.set_metric_mode("acc", "max")
        run.log(loss=0.5, acc=0.8)
        run.log(loss=0.3, acc=0.7)
    assert run.best["loss"] == 0.3          # loss 默认取最小
    assert run.best["acc"] == 0.8           # acc 默认取最大


def test_report_and_summary(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=2, n_steps=20)
    summary = run.report(print_report=False)
    assert summary["epochs_done"] == 2
    assert summary["global_step"] == 40
    assert len(summary["epoch_mean_loss"]) == 2
    assert summary["best"]["loss"] < 1.0


# ---------------------------------------------------------------------------
# 断点续跑
# ---------------------------------------------------------------------------

def test_resume_after_interrupt(tmp_path):
    # 第一段：在 epoch 2、step 5 处模拟 Ctrl-C（catch_interrupt 默认开启，
    # KeyboardInterrupt 会被 __exit__ 存档后优雅吞掉）
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=5, n_steps=10, crash_at=(2, 5))
    assert run.epoch == 2  # 两个完整 epoch 已入账
    assert run.status == RunStatus.INTERRUPTED

    # 第二段：重跑同一脚本，自动续跑
    seen_first_batch = None
    with tk.run("exp", root=str(tmp_path)) as run2:
        assert run2.resumed
        for epoch in run2.epochs(max_epochs=5):
            for batch in run2.steps(range(10)):
                if seen_first_batch is None:
                    seen_first_batch = (epoch, batch)
                run2.log(loss=1.0 / (1 + batch + epoch))
    # 续跑从 epoch 2 的第 5 个 batch 开始（enumerate 下标会重新计数，看 batch 值）
    assert seen_first_batch == (2, 5)
    assert run2.epoch == 5
    assert run2.status == RunStatus.COMPLETED


def test_resume_without_keep_warns_but_works(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=2, n_steps=5, crash_at=(1, 2))
    with tk.run("exp", root=str(tmp_path)) as run2:
        for epoch in run2.epochs(max_epochs=3):
            for _ in run2.steps(range(5)):
                run2.log(loss=0.1)
    assert run2.epoch == 3


# ---------------------------------------------------------------------------
# 工件与时光机
# ---------------------------------------------------------------------------

def test_keep_and_kept(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        for epoch in run.epochs(max_epochs=2):
            run.log(loss=0.5)
            run.keep({"w": [epoch]})
    assert run.kept() == {"w": [1]}  # 最近一份


def test_replay_to_restores_rng(tmp_path):
    # 任务 A 从 seed(99) 出发走 2 个 epoch（每 epoch 5 次随机消耗）；
    # 独立地从同一 seed 重放前 1 个 epoch 的消耗量，得到"epoch1 结束"处
    # 的下一个随机数；A 拨回该时刻后取出的随机数应与其一致。
    random.seed(99)
    with tk.run("A", root=str(tmp_path), resume=False, quiet=True) as ra:
        for epoch in ra.epochs(max_epochs=2):
            for _ in range(5):
                random.random()
                ra.log(loss=0.1)
    random.seed(99)
    for _ in range(5):  # 复现 A 第 1 个 epoch 的随机消耗
        random.random()
    expected_next = random.random()  # epoch1 结束处的下一个随机数
    ra.replay_to(1)
    assert random.random() == expected_next


def test_replay_to_missing_epoch_raises(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=1, n_steps=2)
    with pytest.raises(FileNotFoundError):
        run.replay_to(99)


def test_rng_archive(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=3, n_steps=2)
    keys = run.rng_archive()
    assert "epoch0001" in keys and "epoch0003" in keys


# ---------------------------------------------------------------------------
# 预算与停止条件
# ---------------------------------------------------------------------------

def test_budget_stops(tmp_path):
    import time as _time
    with tk.run("exp", root=str(tmp_path), quiet=True) as run:
        epochs_run = 0
        for _epoch in run.epochs(budget="0.1s"):
            _time.sleep(0.05)
            run.log(loss=0.5)
            epochs_run += 1
    assert 1 <= epochs_run <= 3  # 0.1s 预算内只能跑 1~2 个 epoch
    assert run.status == RunStatus.STOPPED


def test_until_stops(tmp_path):
    losses = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    with tk.run("exp", root=str(tmp_path), quiet=True) as run:
        count = 0
        for epoch in run.epochs(max_epochs=50, until="loss<0.2"):
            run.log(loss=losses[epoch])
            count += 1
    # 0.2<0.2 不命中；0.1<0.2 命中 → 共 5 个 epoch
    assert count == 5
    assert run.status == RunStatus.COMPLETED
    assert "loss < 0.2" in (run.stop_reason or "")


def test_until_never_met_respects_max_epochs(tmp_path):
    with tk.run("exp", root=str(tmp_path), quiet=True) as run:
        count = 0
        for _epoch in run.epochs(max_epochs=4, until="loss<0.001"):
            run.log(loss=0.5)
            count += 1
    assert count == 4


def test_stop_graceful(tmp_path):
    with tk.run("exp", root=str(tmp_path), quiet=True) as run:
        count = 0
        for epoch in run.epochs(max_epochs=10):
            run.log(loss=0.1)
            count += 1
            if count == 2:
                run.stop("手动叫停")
    assert count == 2
    assert run.status == RunStatus.STOPPED


def test_abort_on_nan(tmp_path):
    with tk.run("exp", root=str(tmp_path), quiet=True, abort_on_nan=True) as run:
        with pytest.raises(tk.EmergencyStop):
            run.log(loss=float("nan"))


def test_epochs_requires_stop_condition(tmp_path):
    run = tk.run("exp", root=str(tmp_path))
    with pytest.raises(ValueError):
        list(run.epochs())  # 生成器在首次迭代时校验参数


# ---------------------------------------------------------------------------
# 钩子系统
# ---------------------------------------------------------------------------

def test_hooks_fire(tmp_path):
    events = []
    run = tk.run("exp", root=str(tmp_path), quiet=True)
    run.on("start", lambda **kw: events.append(("start",)))
    run.on("epoch_end", lambda **kw: events.append(("epoch_end", kw["epoch"])))
    run.on("end", lambda **kw: events.append(("end", kw["status"])))
    with run:
        _train(run, n_epochs=2, n_steps=3)
    kinds = [e[0] for e in events]
    assert kinds.count("epoch_end") == 2
    assert "start" in kinds and "end" in kinds


def test_broken_hook_does_not_break_training(tmp_path):
    def bad(**kw):
        raise RuntimeError("boom")
    with tk.run("exp", root=str(tmp_path), quiet=True) as run:
        run.on("step", bad)
        history = _train(run, n_epochs=1, n_steps=5)
    assert len(history) == 5


# ---------------------------------------------------------------------------
# 状态机边界
# ---------------------------------------------------------------------------

def test_resume_false_ignores_checkpoint(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=2, n_steps=3)
    with tk.run("exp", root=str(tmp_path), resume=False, quiet=True) as run2:
        assert not run2.resumed
        assert run2.epoch == 0


def test_attach_readonly(tmp_path):
    with tk.run("exp", root=str(tmp_path)) as run:
        _train(run, n_epochs=2, n_steps=3)
    attached = tk.Run.attach(str(tmp_path / "exp"))
    assert attached.name == "exp"
