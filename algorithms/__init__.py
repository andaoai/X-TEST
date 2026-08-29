"""
算法注册表 —— 加新算法在这里 import
====================================
步骤:
  1. 复制 template.py → algo_xxx.py
  2. 实现 encode()
  3. 在这里 import + 加入 ALGOS
"""
from algorithms.template import MyAlgorithm
from algorithms.color_tokens import ColorTokensAlgo

ALGOS = {
    "random_proj":  MyAlgorithm(),        # 现有基线:整图随机投影
    "color_tokens": ColorTokensAlgo(),    # 新算法:HSV 分解 + CNN token + 聚合(NLP 类比)
}
