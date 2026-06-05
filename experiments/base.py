"""
实验基类 —— 每个实验声明四要素:

  1. 数据来源:  data_source   (数据从哪来，需要什么标签字段)
  2. 实验步骤:  run()         (如何执行度量)
  3. 评判标准:  passes_when   (怎样算"正确")
  4. 可视化:    viz()         (如何把结果变成图)

一个完整的实验文件 = 继承 BaseExperiment + 实现 run() + 实现 viz()
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseExperiment(ABC):
    # ── 四要素声明 ──
    name: str = ""              # 实验名称
    hypothesis: str = ""        # 要验证什么假设
    data_source: str = ""       # 数据从哪来 (e.g. "experiments/source/synthetic")
    what_labels: list[str] = [] # 需要 labels 中哪些字段
    passes_when: str = ""       # 通过条件 (人类可读)

    @abstractmethod
    def run(self, emb: np.ndarray, labels: dict, sim: np.ndarray) -> dict:
        """
        度量 → 返回结果。

        Returns:
            {
                "name": self.name,
                "hypothesis": self.hypothesis,
                "metric": str,           # "separation=+0.16" 或 "sim=0.85"
                "separation": float,     # 核心数值 (越大越好)
                "is_correct": bool,      # 是否正确
                "details": dict,         # 任意额外信息
            }
        """
        ...

    @abstractmethod
    def viz(self, emb: np.ndarray, labels: dict, result: dict,
            algo_name: str, out_dir: Path):
        """把实验结果变成一张图"""
        ...

    def check_labels(self, labels: dict) -> bool:
        for field in self.what_labels:
            if field not in labels:
                print(f"  [WARN] 缺少标签 '{field}', 跳过 {self.name}")
                return False
        return True
