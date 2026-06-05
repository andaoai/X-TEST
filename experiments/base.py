"""
实验基类 —— 每个实验声明五要素:

  1. 数据从哪来:  load_data()   (返回 masks/rgb + labels)
  2. 假设是什么:  hypothesis
  3. 怎么度量:    run()
  4. 怎样算对:    passes_when
  5. 如何可视化:  viz()
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseExperiment(ABC):
    name: str = ""
    hypothesis: str = ""
    passes_when: str = ""
    uses_rgb: bool = True   # True=算法需要RGB, False=算法只需要mask

    @abstractmethod
    def load_data(self):
        """
        加载数据 → (inputs, labels)

        Returns:
            inputs: (N, H, W) mask 或 (N, H, W, 3) RGB
            labels: {field: {value: [indices]}}
        """
        ...

    @abstractmethod
    def run(self, emb: np.ndarray, labels: dict, sim: np.ndarray) -> dict:
        """
        度量 → 返回结果。
        {
            "name": ..., "hypothesis": ..., "metric": str,
            "separation": float, "is_correct": bool, "details": dict,
        }
        """
        ...

    @abstractmethod
    def viz(self, emb: np.ndarray, labels: dict, result: dict,
            algo_name: str, out_dir: Path):
        """可视化"""
        ...

    def check_labels(self, labels: dict, required: list[str]) -> bool:
        for field in required:
            if field not in labels:
                print(f"  [WARN] 缺少标签 '{field}', 跳过 {self.name}")
                return False
        return True
