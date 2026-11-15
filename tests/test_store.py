"""持久层测试：原子写、读回、工件存取。"""
import os

import pytest

from trainkeeper.store import HistoryLog, JsonStateStore, PickleArtifactStore


def test_json_store_roundtrip(tmp_path):
    store = JsonStateStore(str(tmp_path / "run1"))
    assert store.read("state") is None
    assert not store.exists("state")
    store.write("state", {"epoch": 3, "best": {"loss": 0.1}})
    assert store.exists("state")
    assert store.read("state")["epoch"] == 3


def test_json_store_overwrite(tmp_path):
    store = JsonStateStore(str(tmp_path / "run1"))
    store.write("state", {"epoch": 1})
    store.write("state", {"epoch": 2})
    assert store.read("state")["epoch"] == 2
    # 不允许留下临时文件垃圾
    leftovers = [f for f in os.listdir(str(tmp_path / "run1")) if f.endswith(".tmp")]
    assert leftovers == []


def test_artifact_store_roundtrip(tmp_path):
    store = PickleArtifactStore(str(tmp_path / "artifacts"))
    obj = {"weights": [1.0, 2.0], "meta": ("a", 1)}
    path = store.save("epoch0001", obj)
    assert os.path.exists(path)
    assert store.load("epoch0001") == obj
    assert store.load("missing") is None


def test_artifact_store_keys(tmp_path):
    store = PickleArtifactStore(str(tmp_path / "artifacts"))
    store.save("rng/epoch0001", 1)
    store.save("rng/epoch0002", 2)
    keys = list(store.keys())
    assert "rng/epoch0001" in keys and "rng/epoch0002" in keys


def test_history_log_append_read(tmp_path):
    log = HistoryLog(str(tmp_path / "history.jsonl"))
    log.append({"step": 1, "loss": 0.5})
    log.append({"step": 2, "loss": 0.4, "event": "spike"})
    recs = log.read_all()
    assert len(recs) == 2
    assert recs[1]["event"] == "spike"
    # 空文件读回为空列表
    empty = HistoryLog(str(tmp_path / "none.jsonl"))
    assert empty.read_all() == []
