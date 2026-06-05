#!/usr/bin/env python
"""
Mask Embedding Lab · 入口

三部分:
  1. 数据从哪来 → experiments/source/synthetic/ (合成) or real/ (真实)
  2. 算法怎么编码 → algorithms/
  3. 怎么验证对错 → experiments/text/ (每个实验一个文件, 四要素完整)

用法:
  uv run python run.py                          # 全部
  uv run python run.py --algo gabor_lift        # 选算法
  uv run python run.py --exp exp3               # 单实验
  uv run python run.py --letters A,B,C          # 自定义字母
  uv run python run.py --colors Red,Blue        # 自定义颜色
  uv run python run.py --no-viz                 # 跳过图片
"""
import sys, json, argparse
from pathlib import Path

_here = Path(__file__).parent.resolve()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import numpy as np
from algorithms import ALGOS
from experiments.source.synthetic.config import OUTPUT_ROOT, COLORS
from experiments.source.synthetic.data import Dataset
from experiments.text import TEXT_EXPERIMENTS
from experiments.viz import summary_bar


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def parse_args():
    p = argparse.ArgumentParser("Mask Embedding Lab")
    p.add_argument("--algo", default="gabor_lift", choices=list(ALGOS))
    p.add_argument("--exp",  default=None, choices=["exp1","exp2","exp3","exp4","exp5"])
    p.add_argument("--letters", default=None, help="A,B,C")
    p.add_argument("--colors",  default=None, help="Red,Blue")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    algo = ALGOS[args.algo]

    print("=" * 55)
    print(f"  Mask Embedding Lab  |  算法: {algo.name}")
    print("=" * 55)

    # ═══ 1. 数据: 从哪里来 ═══
    print("\n[1/3] 数据来源: experiments/source/synthetic/data.py")
    kw = {}
    if args.letters:
        kw["letters"] = [x.strip() for x in args.letters.split(",")]
        kw["chinese"] = []
    if args.colors:
        sel = [x.strip() for x in args.colors.split(",")]
        kw["colors"] = {c: COLORS[c] for c in sel if c in COLORS}
    ds = Dataset(**kw).generate()

    # ═══ 2. 编码: 算法 mask → embedding ═══
    print(f"\n[2/3] 算法: algorithms/{algo.name}.py")
    inputs = ds.rgbs() if algo.uses_rgb else ds.masks()
    emb = algo.encode(inputs)
    sim = emb @ emb.T

    # ═══ 3. 实验: 验证假设 + 评判 + 可视化 ═══
    print(f"\n[3/3] 实验: experiments/text/")
    results = {}
    to_run = [args.exp] if args.exp else list(TEXT_EXPERIMENTS.keys())

    for exp_key in to_run:
        exp = TEXT_EXPERIMENTS[exp_key]
        if not exp.check_labels(ds.labels):
            continue
        r = exp.run(emb, ds.labels, sim)
        r["is_correct"] = bool(r["is_correct"])
        results[exp_key] = r
        tag = "OK" if r["is_correct"] else "NO"
        print(f"  [{tag}] {exp.name:12s}  {r['metric']}")

    # 保存指标
    out = OUTPUT_ROOT / algo.name
    out.mkdir(parents=True, exist_ok=True)
    metrics = {k: _json_safe({"is_correct": r["is_correct"], **r["details"]})
               for k, r in results.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ═══ 4. 可视化 ═══
    if not args.no_viz:
        print("\n  可视化...")
        for exp_key, r in results.items():
            exp = TEXT_EXPERIMENTS.get(exp_key)
            if exp and hasattr(exp, "viz"):
                exp.viz(emb, ds.labels, r, algo.name, out)
        # 汇总
        summary_bar(results, algo.name, out / "summary.png")
        print(f"   图表 → {out}")

    print(f"\n完成 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
