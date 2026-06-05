#!/usr/bin/env python
"""
Mask Embedding Lab

三部分:
  1. 数据从哪来 → 每个实验自己提供 load_data()
  2. 算法怎么编码 → algorithms/
  3. 怎么验证对错 → experiments/{text,vision}/  (每个实验=一个文件)

用法:
  uv run python run.py                          # 默认文字实验
  uv run python run.py --algo gabor_lift
  uv run python run.py --exp pixel              # 像素实验
  uv run python run.py --category vision        # 跑全部视觉实验
"""
import sys, json, argparse
from pathlib import Path

_here = Path(__file__).parent.resolve()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np
from algorithms import ALGOS
from experiments.source.synthetic.config import OUTPUT_ROOT, COLORS
from experiments.text import TEXT_EXPERIMENTS
from experiments.text.independence import INDEPENDENCE_EXPERIMENTS
from experiments.text.separability import SEPARABILITY_EXPERIMENTS
from experiments.text.capacity import CAPACITY_EXPERIMENTS
from experiments.vision import VISION_EXPERIMENTS
from experiments.vision.independence import VISION_INDEPENDENCE_EXPERIMENTS
from experiments.vision.separability import VISION_SEPARABILITY_EXPERIMENTS
from experiments.vision.capacity import VISION_CAPACITY_EXPERIMENTS
from experiments.viz import summary_bar

ALL_EXPERIMENTS = {
    **TEXT_EXPERIMENTS,
    **VISION_EXPERIMENTS,
    **{k: v() for k, v in INDEPENDENCE_EXPERIMENTS.items()},
    **{k: v() for k, v in SEPARABILITY_EXPERIMENTS.items()},
    **{k: v() for k, v in CAPACITY_EXPERIMENTS.items()},
    **{k: v() for k, v in VISION_INDEPENDENCE_EXPERIMENTS.items()},
    **{k: v() for k, v in VISION_SEPARABILITY_EXPERIMENTS.items()},
    **{k: v() for k, v in VISION_CAPACITY_EXPERIMENTS.items()},
}


def _json_safe(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_json_safe(v) for v in obj]
    return obj


def parse_args():
    p = argparse.ArgumentParser("Mask Embedding Lab")
    p.add_argument("--algo", default="random_proj", choices=list(ALGOS))
    p.add_argument("--exp",  default=None, metavar="NAME",
                   help=f"单实验: {', '.join(ALL_EXPERIMENTS)}")
    p.add_argument("--category", default=None,
                   choices=["text", "vision", "independence", "separability", "capacity",
                            "vi_independence", "vi_separability", "vi_capacity"])
    p.add_argument("--list", action="store_true", help="列出所有实验")
    p.add_argument("--letters", default=None, help="A,B,C (仅文字)")
    p.add_argument("--colors",  default=None, help="Red,Blue (仅文字)")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    algo = ALGOS[args.algo]

    # 列出实验
    if args.list:
        print("\n实验列表:")
        for key, exp in ALL_EXPERIMENTS.items():
            print(f"  {key:14s} [{exp.__class__.__module__.split('.')[-2]:6s}] {exp.name}")
        return 0

    # 选实验
    if args.exp:
        if args.exp not in ALL_EXPERIMENTS:
            print(f"未知实验: {args.exp}. 可用: {', '.join(ALL_EXPERIMENTS)}")
            return 1
        exps = {args.exp: ALL_EXPERIMENTS[args.exp]}
    elif args.category == "vision":
        exps = dict(VISION_EXPERIMENTS)
    elif args.category == "text":
        exps = dict(TEXT_EXPERIMENTS)
    elif args.category == "independence":
        exps = {k: v() for k, v in INDEPENDENCE_EXPERIMENTS.items()}
    elif args.category == "separability":
        exps = {k: v() for k, v in SEPARABILITY_EXPERIMENTS.items()}
    elif args.category == "capacity":
        exps = {k: v() for k, v in CAPACITY_EXPERIMENTS.items()}
    elif args.category == "vi_independence":
        exps = {k: v() for k, v in VISION_INDEPENDENCE_EXPERIMENTS.items()}
    elif args.category == "vi_separability":
        exps = {k: v() for k, v in VISION_SEPARABILITY_EXPERIMENTS.items()}
    elif args.category == "vi_capacity":
        exps = {k: v() for k, v in VISION_CAPACITY_EXPERIMENTS.items()}
    else:
        exps = dict(TEXT_EXPERIMENTS)  # 默认文字

    print("=" * 55)
    print(f"  Mask Embedding Lab  |  算法: {algo.name}")
    print(f"  实验: {list(exps.keys())}")
    print("=" * 55)

    results = {}
    out = None

    for exp_key, exp in exps.items():
        print(f"\n── {exp.name} ──")

        # 1. 实验自己加载数据
        print(f"  [1/3] 数据: {exp.__class__.__name__}.load_data()")
        inputs, labels = exp.load_data()

        # 2. 编码
        print(f"  [2/3] 编码: {algo.name}")
        emb = algo.encode(inputs)
        sim = emb @ emb.T

        # 3. 度量
        print(f"  [3/3] 实验...")
        r = exp.run(emb, labels, sim)
        r["is_correct"] = bool(r.get("is_correct", False))
        results[exp_key] = r
        tag = "OK" if r["is_correct"] else "NO"
        print(f"  [{tag}] {exp.name}  {r.get('metric','?')}")

        # 保存
        out = OUTPUT_ROOT / algo.name
        out.mkdir(parents=True, exist_ok=True)
        metrics = {k: _json_safe({"is_correct": v["is_correct"], **v.get("details",{})})
                   for k, v in results.items()}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        # 可视化
        if not args.no_viz:
            exp.viz(emb, labels, r, algo.name, out)

    # 汇总
    if not args.no_viz and len(results) > 1:
        summary_bar(results, algo.name, out / "summary.png")

    print(f"\n完成 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
