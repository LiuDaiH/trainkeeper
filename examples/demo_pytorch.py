"""trainkeeper + PyTorch 完整示例：断点续跑一个真实的回归训练。

要点：
- run.keep(model.state_dict()) 每个 epoch 存权重；
- 续跑时 run.kept() 取回最近权重并 load_state_dict；
- budget="20s" 演示到点收工；中途 Ctrl-C 后重跑本脚本可从断点继续。

运行::

    python examples/demo_pytorch.py            # 全新训练
    python examples/demo_pytorch.py            # 再次运行 = 自动续跑
"""
import trainkeeper as tk

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("本示例需要 PyTorch：pip install torch")

# 固定种子——配合 Ledger，断点续跑后数据顺序也完全一致
torch.manual_seed(42)

# 一个刻意过小、容易过拟合出"平台期"的玩具模型
model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

with tk.run("pytorch-demo", abort_on_nan=True) as run:
    # 断点续跑：恢复上次的权重与优化器状态
    snapshot = run.kept()
    if snapshot is not None:
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        print("已恢复上次训练的权重")

    # 把 torch 的 GPU 生成器也纳入时光机（CPU 端默认已覆盖）
    if torch.cuda.is_available():
        run.ledger.register(type("CudaRNG", (), {
            "name": "torch.cuda",
            "get_state": staticmethod(torch.cuda.get_rng_state),
            "set_state": staticmethod(torch.cuda.set_rng_state),
        })())

    for epoch in run.epochs(max_epochs=500, budget="20s", until="loss<0.01"):
        for batch in run.steps(range(64)):  # 64 个合成 batch
            x = torch.randn(32, 4)
            y = (x ** 2).sum(dim=1, keepdim=True) / 4
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run.log(loss=loss.item())

        run.keep({"model": model.state_dict(), "optimizer": optimizer.state_dict()})

    run.report()
