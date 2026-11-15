"""持久层：Run 状态与工件的读写抽象。

设计要点：
- 所有写操作都是**原子写**（先写临时文件再 ``os.replace``），进程在任何时刻被
  杀掉都不会留下半截状态文件。
- 持久层通过 Protocol 定义接口，默认实现为本地 JSON / pickle 文件。实现同名
  协议即可把状态迁移到 Redis、S3 或数据库，训练代码零改动。
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
from typing import Any, Dict, Optional, Protocol, runtime_checkable

__all__ = ["StateStore", "ArtifactStore", "JsonStateStore", "PickleArtifactStore", "HistoryLog"]


def _atomic_replace(tmp: str, path: str, retries: int = 5, delay: float = 0.02) -> None:
    """os.replace，附 Windows 瞬态锁重试。

    Windows 上杀毒软件 / 文件索引服务可能短暂锁住刚创建的文件导致
    PermissionError；这类锁是毫秒级的，短暂重试即可恢复。
    """
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Protocols（接口契约）
# ---------------------------------------------------------------------------

@runtime_checkable
class StateStore(Protocol):
    """Run 状态的键值式存储接口。"""

    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """读取状态；不存在时返回 None。"""
        ...

    def write(self, key: str, state: Dict[str, Any]) -> None:
        """原子写入状态。"""
        ...

    def exists(self, key: str) -> bool:
        """状态是否存在。"""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """任意 Python 对象（模型权重、优化器状态、RNG 快照等）的存取接口。"""

    def save(self, key: str, obj: Any) -> str:
        """保存对象，返回存储路径。"""
        ...

    def load(self, key: str) -> Optional[Any]:
        """读取对象；不存在时返回 None。"""
        ...

    def keys(self) -> Any:
        """列出全部已存储的 key（可迭代）。"""
        ...


# ---------------------------------------------------------------------------
# 默认实现
# ---------------------------------------------------------------------------

class JsonStateStore:
    """基于本地 JSON 文件的 :class:`StateStore` 实现。"""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, "{}.json".format(key))

    def read(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self._path(key), "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def write(self, key: str, state: Dict[str, Any]) -> None:
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, default=str)
            _atomic_replace(tmp, path)  # 同目录原子替换
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class PickleArtifactStore:
    """基于 pickle 的 :class:`ArtifactStore` 实现。

    torch 张量 / state_dict 均可直接 pickle，无需 torch.save。
    """

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_")
        return os.path.join(self.root, safe + ".pkl")

    def save(self, key: str, obj: Any) -> str:
        path = self._path(key)
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            _atomic_replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return path

    def load(self, key: str) -> Optional[Any]:
        try:
            with open(self._path(key), "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def keys(self):
        for fname in sorted(os.listdir(self.root)):
            if fname.endswith(".pkl"):
                yield fname[:-4].replace("_", "/")


class HistoryLog:
    """追加式事件/指标日志（JSON Lines）。仅追加、从不重写，天然崩溃安全。"""

    def __init__(self, path: str) -> None:
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self):
        records = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records
