"""诊断层测试：内置规则与 Doctor 聚合。"""
import pytest

from trainkeeper.doctor import (
    Diagnosis,
    Doctor,
    DivergenceRule,
    NaNLossRule,
    PlateauRule,
    SpikeRule,
)


def test_nan_rule_fires():
    rule = NaNLossRule(cooldown=1)
    hits = rule.observe(10, {"loss": float("nan")})
    assert len(hits) == 1
    assert hits[0].level == "error"
    assert hits[0].kind == "nan_loss"
    rule2 = NaNLossRule(cooldown=1)
    assert rule2.observe(11, {"loss": float("inf")})


def test_nan_rule_silent_on_healthy():
    rule = NaNLossRule()
    assert rule.observe(1, {"loss": 0.5}) == []
    assert rule.observe(2, {}) == []  # 没有 loss 键时不误报


def test_divergence_rule_fires_on_rising_loss():
    rule = DivergenceRule(window=50, threshold=0.005, cooldown=10**9)
    hits = []
    for i in range(60):
        hits += rule.observe(i, {"loss": 1.0 + i * 0.02})  # 每步 +2%
    assert len(hits) >= 1
    assert hits[0].kind == "divergence"
    assert hits[0].level == "warn"


def test_divergence_rule_silent_on_falling_loss():
    rule = DivergenceRule(window=50)
    hits = []
    for i in range(100):
        hits += rule.observe(i, {"loss": 1.0 - i * 0.005})
    assert hits == []


def test_spike_rule_fires():
    rule = SpikeRule(window=50, factor=6.0, cooldown=10**9)
    hits = []
    for i in range(60):
        value = 1.0 + (50.0 if i == 55 else 0.0)
        hits += rule.observe(i, {"loss": value})
    assert len(hits) == 1
    assert hits[0].kind == "spike"


def test_plateau_rule_fires_on_flat_loss():
    rule = PlateauRule(window=150, tolerance=0.02, cooldown=10**9)
    hits = []
    for i in range(160):
        hits += rule.observe(i, {"loss": 0.5 + (i % 3) * 0.0005})
    assert len(hits) >= 1
    assert hits[0].kind == "plateau"
    assert hits[0].level == "info"


def test_cooldown_prevents_spam():
    rule = DivergenceRule(window=50, cooldown=100)
    fired = 0
    for i in range(200):
        fired += len(rule.observe(i, {"loss": 1.0 + i * 0.02}))
    assert fired == 2  # 在 step≈49 和 149 各报一次，冷却期内不刷屏


def test_doctor_aggregates_rules():
    doctor = Doctor(rules=[NaNLossRule(), DivergenceRule(window=50, cooldown=10**9)])
    hits = doctor.observe(0, {"loss": float("nan")})
    assert any(d.kind == "nan_loss" for d in hits)
    hits = doctor.observe(0, {"acc": 0.9})  # 无 loss 时安静
    assert hits == []


def test_doctor_survives_faulty_rule():
    class Bad:
        name = "bad"
        def observe(self, step, metrics):
            raise RuntimeError("boom")

    doctor = Doctor(rules=[NaNLossRule(), Bad()])
    hits = doctor.observe(1, {"loss": float("nan")})  # 坏规则不影响好规则
    assert any(d.kind == "nan_loss" for d in hits)


def test_custom_rule_via_protocol():
    class AlwaysFire:
        name = "always"
        def observe(self, step, metrics):
            return [Diagnosis("info", "always", "m", "a", step)]

    doctor = Doctor(rules=[])
    doctor.add_rule(AlwaysFire())
    assert doctor.rule_names == ["always"]
    assert len(doctor.observe(1, {})) == 1


def test_diagnosis_format():
    d = Diagnosis("warn", "spike", "msg", "advice", 42)
    text = d.format()
    assert "警告" in text and "msg" in text and "advice" in text
