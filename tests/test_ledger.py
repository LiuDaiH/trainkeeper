"""可复现层测试：RNG 快照/恢复的精确性。"""
import random

import pytest

from trainkeeper.ledger import Ledger, StdLibRandomSource


def _draw(n=5):
    return [random.random() for _ in range(n)]


def test_random_module_replay():
    ledger = Ledger(include_defaults=False)
    ledger.register(StdLibRandomSource())
    random.seed(1234)
    snap = ledger.snapshot()
    first = _draw()
    random.seed(999)
    _draw()  # 消耗随机数，污染状态
    ledger.restore(snap)
    second = _draw()
    assert first == second


def test_default_sources_include_random():
    ledger = Ledger()
    assert "random" in ledger.source_names()


def test_numpy_replay_if_available():
    np = pytest.importorskip("numpy")
    ledger = Ledger()
    np.random.seed(7)  # 先定种子再快照
    snap = ledger.snapshot()
    a = np.random.rand(4).tolist()
    np.random.rand(3)  # 污染
    ledger.restore(snap)
    b = np.random.rand(4).tolist()
    assert a == b


def test_register_custom_source():
    class Counter:
        name = "counter"
        def __init__(self):
            self.value = 0
        def get_state(self):
            return self.value
        def set_state(self, state):
            self.value = state

    ledger = Ledger(include_defaults=False)
    c = Counter()
    ledger.register(c)
    assert "counter" in ledger.source_names()
    c.value = 42
    snap = ledger.snapshot()
    c.value = 100
    ledger.restore(snap)
    assert c.value == 42


def test_restore_partial():
    ledger = Ledger(include_defaults=False)
    ledger.register(StdLibRandomSource())
    random.seed(1)
    snap = ledger.snapshot()
    expected = _draw()  # seed(1) 状态下的前 5 个随机数
    random.seed(2)
    _draw()
    ledger.restore_partial(snap, names=[])  # 什么都不恢复，流仍是 seed(2) 的
    assert _draw() != expected
    ledger.restore_partial(snap, names=["random"])  # 只恢复 random
    assert _draw() == expected
