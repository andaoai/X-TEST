"""
算法注册表 —— 加新算法在这里 import
====================================
步骤:
  1. 复制 template.py → algo_xxx.py
  2. 实现 encode()
  3. 在这里 import + 加入 ALGOS
"""
from algorithms.gabor import GaborLift
from algorithms.template import MyAlgorithm

ALGOS = {
    "gabor_lift": GaborLift(),
    "random_proj": MyAlgorithm(),
}
