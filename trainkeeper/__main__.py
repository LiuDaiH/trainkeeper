"""trainkeeper CLI。

用法::

    tk list                  # 列出 .tkruns 下的全部任务
    tk report <run目录|任务名>   # 查看训练事后报告
    tk inspect <run目录>      # 查看原始状态 JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .run import format_summary, load_summary


def _resolve_run_dir(target: str) -> str:
    """支持传任务名（相对 .tkruns）或直接传目录。"""
    if os.path.isdir(target):
        return target
    guess = os.path.join(".tkruns", target)
    if os.path.isdir(guess):
        return guess
    raise SystemExit("找不到任务目录：{}（尝试过 {}）".format(target, guess))


def _find_runs(root: str = ".tkruns") -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tk",
        description="trainkeeper CLI：查看训练任务的状态与报告。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出全部训练任务")
    p_list.add_argument("--root", default=".tkruns")

    p_report = sub.add_parser("report", help="查看某个任务的训练报告")
    p_report.add_argument("target", help="任务名或 run 目录")

    p_inspect = sub.add_parser("inspect", help="查看原始状态 JSON")
    p_inspect.add_argument("target")

    args = parser.parse_args(argv)

    if args.command == "list":
        runs = _find_runs(args.root)
        if not runs:
            print("（{} 下没有任务）".format(args.root))
            return 0
        for name in runs:
            summary = load_summary(os.path.join(args.root, name))
            print("{:<24} epochs={:<4} steps={:<8} status={}".format(
                name, summary["epochs_done"], summary["global_step"],
                summary["state"].get("status", "?")))
        return 0

    if args.command == "report":
        run_dir = _resolve_run_dir(args.target)
        print(format_summary(load_summary(run_dir)))
        return 0

    if args.command == "inspect":
        run_dir = _resolve_run_dir(args.target)
        from .store import JsonStateStore
        state = JsonStateStore(run_dir).read("state")
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
