"""trainkeeper 快速上手示例（零依赖，任何 Python 3.8+ 环境可直接运行）。

本示例模拟一个真实训练循环，展示 trainkeeper 的五个核心能力：

1. 断点续跑：训练中按 Ctrl-C，重跑本脚本即从断点继续；
2. 预算控制：budget="10s" 时间到优雅收工；
3. 在线诊断：loss 出现 NaN / 发散 / 尖峰时，医生当场报警并给建议；
4. RNG 时光机：每个 epoch 自动存档，replay_to 可拨回任意时刻；
5. 事后报告：run.report() 输出完整训练报告。

运行::

    python examples/demo_quickstart.py
"""
import argparse
import random
import time

import trainkeeper as tk


def fake_training_step(epoch: int, step: int, rng: random.Random) -> float:
    """伪造一个 loss 函数，埋入典型故障模式，方便观察医生的反应。"""
    base = 1.0 / (1.0 + epoch * 0.5 + step * 0.01)
    noise = rng.gauss(0, 0.02)
    # 第 3 个 epoch 的第 30 步埋一次 NaN（abort_on_nan=False 时医生只报警不打断）
    if epoch == 2 and step == 30:
        return float("nan")
    # 第 5 个 epoch 的第 20 步埋一次尖峰
    if epoch == 4 and step == 20:
        return base + 8.0
    return max(1e-6, base + noise)


def main() -> None:
    parser = argparse.ArgumentParser(description="trainkeeper 快速上手示例")
    parser.add_argument("--budget", default="10s", help="时间预算，如 10s / 5m")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    rng = random.Random(2026)

    # with 一句包住你的训练循环，其余代码不变
    with tk.run("quickstart-demo", quiet=args.quiet, abort_on_nan=False) as run:
        # 事件钩子示例：每个 epoch 结束打印一次当前最优
        run.on("epoch_end", lambda **kw: print(
            "  └─ epoch {} 完成，当前最优 loss {:.4f}".format(
                kw["epoch"], run.best.get("loss", float("nan")))))

        for epoch in run.epochs(max_epochs=100, budget=args.budget):
            for step in run.steps(range(60)):
                loss = fake_training_step(epoch, step, rng)
                # log() 是心跳：记录指标 + 触发医生体检
                run.log(loss=loss, lr=0.001 * (0.5 ** (epoch // 5)))
                time.sleep(0.002)  # 模拟真实计算耗时

            # 每个 epoch 保存一次"权重"（任何可 pickle 对象都行）
            run.keep({"epoch": epoch, "weights": [rng.random() for _ in range(4)]})

            # 时光机演示：第 6 个 epoch 结束时拨回第 3 个 epoch 结束的那一刻
            if epoch == 5:
                run.replay_to(3)
                print("  └─ RNG 已拨回 epoch 3 结束时刻（可复现性验证）")

    print()
    run.report()


if __name__ == "__main__":
    main()
