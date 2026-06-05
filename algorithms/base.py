"""
算法基类。实现新算法只需继承此类并实现 encode()。

接口:
  inputs → encode() → (N, 128) L2归一化 embedding
"""
import numpy as np
from abc import ABC, abstractmethod
EMBEDDING_DIM = 128  # 目标嵌入维度，算法和实验都依赖此常量


class BaseAlgorithm(ABC):
    name: str = "base"
    uses_rgb: bool = False   # True=输入RGB三通道, False=输入二值mask

    @abstractmethod
    def encode(self, inputs: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        inputs: (N, H, W) mask 或 (N, H, W, 3) RGB
        returns: (N, EMBEDDING_DIM)  L2 归一化后的向量
        """
        ...

    def __repr__(self):
        return f"<{self.name}>"
