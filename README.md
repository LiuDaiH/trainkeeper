# trainkeeper

**给 Python AI 训练加上"管家"的基础设施层。**

*A zero-dependency infrastructure layer for AI training in Python: crash-safe resumption, RNG time-travel, online loss diagnosis, and budget control — wrapped around any training loop.*

---

## 为什么需要它

每个写过深度学习的人都遇到过这些问题：

- 训练跑了 6 小时，进程崩了 → **从头再来**
- "为什么这次结果和上次不一样？" → **随机性散落在 random / numpy / torch / DataLoader 里，没人能说清**
- loss 突然 NaN / 突然飞了 / 突然躺平 → **翻 terminal 才发现，错过了最佳抢救时机**
- "再训 5 分钟就收工" → **写 `while time.time() < deadline` 各种补丁**

trainkeeper 用一个上下文管理器解决以上全部问题，**零第三方依赖**，PyTorch / NumPy / 手写循环通吃。

## 安装

```bash
pip install -e .          # 从本仓库安装
# 或发布后：
pip install trainkeeper
```

## 5 分钟上手

**之前：**

```python
for epoch in range(100):
    for batch in loader:
        loss = train_one(batch)
    if time.time() > deadline: break   # 手写预算
    torch.save(...)                     # 手写存档
    # 崩了？重头再来
```

**之后：**

```python
import trainkeeper as tk

with tk.run("exp1") as run:
    for epoch in run.epochs(max_epochs=100, budget="10m", until="loss<0.05"):
        for batch in run.steps(loader):        # 崩溃后自动从断点 batch 继续
            loss = train_one(batch)
            run.log(loss=loss)                  # 心跳：记录 + 在线诊断
        run.keep(model.state_dict())            # 权重档案（续跑时自动可用）
    run.report()                                # 训练结束的完整报告
```

崩了？**重跑同一个脚本**。trainkeeper 自动从断点继续：epoch 级恢复权重（`run.kept()`），step 级恢复进度与随机状态。

```
[trainkeeper] 检测到断点，任务 'exp1' 从 epoch 47 (step 128) 续跑。
e48/100 | s23410 | loss 0.5231 | lr 0.001  ▁▂▂▃▂▃▄▄▅▅▆▆▇
[警告] loss 近 50 步相对斜率为 +0.73%/步，呈发散趋势。
       典型发散信号：降低学习率、开启梯度裁剪；或用 run.replay_to() 回退到上一个稳定 epoch。
```

## 核心能力

### 1. 断点续跑（崩溃安全）

- 状态经**原子写**落盘（先写临时文件再 `os.replace`，带 Windows 瞬态锁重试）；
- 进程被 kill -9、断电、Ctrl-C，进度都不丢；
- `KeyboardInterrupt` 默认被优雅接管：存档、打印断点位置、干净退出（`catch_interrupt=True`）；
- 存档目录 `.tkruns/<任务名>/`：`state.json`（进度）、`history.jsonl`（全量事件流，只追加不重写）、`artifacts/`（权重 + RNG 快照）。

### 2. RNG 时光机（可复现性）

每个 epoch 结束，`Ledger` 自动快照全部已注册随机源（`random`、`numpy.random`、`torch`，以及你通过 `RandomSource` 协议注册的任何自定义随机状态）：

```python
run.replay_to(23)   # 把"整个世界的随机状态"拨回 epoch 23 结束那一刻
```

之后消耗随机数的代码（数据打乱、增强、探索）将与该 epoch 之后的训练轨迹**完全一致**。这是实验复现的关键开关，也是发散后"回档重来"的后悔药。

### 3. 训练医生（在线诊断）

`log()` 即体检。内置四条可插拔规则，命中时用大白话告诉你**现象 + 处置建议**：

| 规则 | 检测 | 级别 |
|---|---|---|
| `NaNLossRule` | loss 出现 NaN/Inf | error |
| `DivergenceRule` | loss 持续上升（相对斜率超阈值） | warn |
| `SpikeRule` | loss 孤立尖峰（远超中位数） | warn |
| `PlateauRule` | loss 长期停滞（波动低于阈值） | info |

全部规则带冷却机制，绝不刷屏。自定义规则只需实现 `DiagnosisRule` 协议：

```python
class MyRule:                       # 实现 observe(step, metrics) -> List[Diagnosis]
    name = "val_gap"
    def observe(self, step, metrics):
        if abs(metrics.get("loss", 0) - metrics.get("val_loss", 0)) > 0.5:
            return [tk.Diagnosis("warn", "val_gap", "过拟合征兆", "加正则或早停", step)]
        return []

run.doctor.add_rule(MyRule())
```

### 4. 预算控制

```python
run.epochs(max_epochs=100,       # 总 epoch 目标（含续跑前已完成）
           budget="10m",          # 时间预算（跨续跑累计，防止总时长失控）
           until="val_acc>=0.9")  # 指标条件（在 epoch 边界判定）
run.stop()                        # 当前 epoch 跑完后优雅收工
run.abort("GPU 温度过高")          # 立即终止，抛出 EmergencyStop
```

### 5. 可观测

- **终端火花线**：不装任何绘图库，loss 曲线直接画在命令行（`▁▂▃▄▅▆▇█`，GBK 终端自动退化 ASCII）；
- **事件流**：全量指标与诊断写入 `history.jsonl`，事后可完整回放；
- **报告**：`run.report()` 输出训练摘要 + 每 epoch 曲线 + 诊断时间线；
- **CLI**：

```bash
tk list                    # 列出全部任务
tk report exp1             # 事后查看某任务的报告
tk inspect .tkruns/exp1    # 原始状态 JSON
```

### 6. 事件钩子

```python
run.on("epoch_end", lambda **kw: maybe_validate(kw["epoch"]))
run.on("diagnosis", lambda **kw: webhook_notify(kw["diagnosis"].format()))
# 事件：start / resume / step / epoch_end / diagnosis / end
```

## API 一览

| 接口 | 说明 |
|---|---|
| `tk.run(name, **cfg)` | 构造 Run（上下文管理器） |
| `run.epochs(max_epochs=None, budget=None, until=None)` | epoch 主循环（三者至少给一） |
| `run.steps(iterable, resume="skip")` | step 子循环（断点步级跳过） |
| `run.log(**metrics)` | 心跳：记录指标、触发诊断，返回 `List[Diagnosis]` |
| `run.keep(obj) / run.kept()` | 保存 / 取回工件（权重、优化器状态等任意可 pickle 对象） |
| `run.replay_to(epoch)` | RNG 时光机 |
| `run.rng_archive()` | 列出全部 RNG 快照 |
| `run.stop(reason) / run.abort(reason)` | 优雅停止 / 立即终止 |
| `run.set_metric_mode(key, "min"/"max")` | 显式指定指标优化方向（默认启发式：含 loss/err 取最小） |
| `run.on(event, fn)` | 事件钩子 |
| `run.report(print=True)` | 事后报告（打印 + 返回结构化 dict） |
| `run.doctor / run.ledger` | 直达诊断引擎 / 随机源注册表 |

**语义约定**：`run.epoch` 是已完成的 epoch 数（0 起）；`max_epochs` 为含续跑的总目标；`until` 依据最近一次 log 值在 epoch 边界判定；`budget` 跨续跑累计。

## 架构

```
trainkeeper/
├── run.py       # 核心层：Run 状态机、生命周期、编排
├── store.py     # 持久层：StateStore / ArtifactStore 协议（默认 JSON + pickle，原子写）
├── ledger.py    # 可复现层：RandomSource 协议、快照/恢复引擎
├── doctor.py    # 诊断层：DiagnosisRule 协议、四条内置规则
├── spark.py     # 可观测层：终端渲染、火花线、格式化
└── __main__.py  # CLI：tk list / report / inspect
```

全部核心组件基于 `typing.Protocol` 定义接口契约（`py.typed`，完整类型注解）：

| 协议 | 默认实现 | 替换示例 |
|---|---|---|
| `StateStore` | `JsonStateStore` | Redis / S3 状态后端 |
| `ArtifactStore` | `PickleArtifactStore` | 云端权重档案 |
| `RandomSource` | random / numpy / torch | DataLoader 采样器、CUDA 生成器 |
| `DiagnosisRule` | 四条内置规则 | 过拟合检测、梯度范数监控 |

```python
run = tk.run("exp", state_store=MyRedisStore(), artifact_store=MyS3Store())
```

## 设计原则

1. **不接管训练循环**——你保留原有代码结构，trainkeeper 只做周边基础设施；
2. **零强制依赖**——`pip install trainkeeper` 不带任何第三方包，numpy/torch 存在则自动接入；
3. **永不打断训练**——渲染、钩子、诊断规则出错一律静默降级；
4. **诚实语义**——续跑的权重恢复粒度、budget 的累计语义都在文档写明，不留隐式惊喜。

## 与现有工具的关系

| 工具 | 重叠 | trainkeeper 的差异化 |
|---|---|---|
| TensorBoard / W&B | 指标可视化 | 不做重型面板；可视化只是副产品，核心是**续跑 + 复现 + 诊断** |
| Lightning | 接管训练循环 | 不接管循环；任何循环（含 Lightning 内部）都可套一层 |
| PyTorch `torch.save` | 手工 checkpoint | 自动化 + 原子写 + 步级断点 + RNG 时光机 |

## 运行测试

```bash
pip install -e .[dev]
pytest
# 42 passed
```

## 示例

```bash
python examples/demo_quickstart.py    # 零依赖快速上手（含 NaN/尖峰/发散 故障注入）
python examples/demo_pytorch.py       # PyTorch 完整示例（断点续跑 + 权重档案）
```

## Roadmap

- [ ] TensorBoard / W&B 渲染后端（Renderer 协议已就绪）
- [ ] CUDA RNG 自动纳入 Ledger
- [ ] 多任务对比报告（`tk compare exp1 exp2`）
- [ ] 诊断规则包（过拟合 / 梯度范数 / 学习率敏感性）

## 许可证

MIT © 2026 liuDH
