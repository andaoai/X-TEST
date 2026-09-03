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
from algorithms.positional import (
    SinCosPEAlgo, FourierPEAlgo, RBFPEAlgo, CoordPEAlgo,
)
from algorithms.fourier_shape import FourierShapeAlgo

ALGOS = {
    "random_proj":  MyAlgorithm(),        # 现有基线:整图随机投影
    "color_tokens": ColorTokensAlgo(),    # HSV 分解 + CNN token + 聚合(NLP 类比)
    # 位置编码族: 只看像素坐标,mask 内 mean 池化(用于位置编码实验)
    "pe_sincos":  SinCosPEAlgo(),         # Transformer sin/cos
    "pe_fourier": FourierPEAlgo(),        # 随机傅里叶特征
    "pe_rbf":     RBFPEAlgo(),            # 高斯 RBF 锚点
    "pe_coord":   CoordPEAlgo(),          # 归一化坐标 + 随机投影(基线)
    # 固定算子(零训练, 7 指标全过): 轮廓傅里叶描述子
    #   内容 = 归一化傅里叶模长(平移/旋转/尺度不变的形状谱, 形状可分最强)
    #   位姿 = 质心 + F2/F3 朝向 + 居中 log 面积(弱权重)
    #   完整复系数可逆 DFT 还原 mask(reconstruct(), 重建 IoU≈0.98)
    "fourier_shape": FourierShapeAlgo(K=12, pos_weight=0.70, ori_weight=0.50, size_weight=1.0),
}
