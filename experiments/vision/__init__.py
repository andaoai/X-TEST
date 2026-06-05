"""视觉实验 —— 控制变量实验"""
from experiments.vision.pixel import ExpPixelSeparation

# 控制变量实验
from experiments.vision.independence import VISION_INDEPENDENCE_EXPERIMENTS
from experiments.vision.separability import VISION_SEPARABILITY_EXPERIMENTS
from experiments.vision.capacity import VISION_CAPACITY_EXPERIMENTS

VISION_EXPERIMENTS = {
    # 旧实验（保留兼容）
    "pixel": ExpPixelSeparation(),

    # 新实验：独立性
    **{k: v() for k, v in VISION_INDEPENDENCE_EXPERIMENTS.items()},

    # 新实验：可分性
    **{k: v() for k, v in VISION_SEPARABILITY_EXPERIMENTS.items()},

    # 新实验：容量
    **{k: v() for k, v in VISION_CAPACITY_EXPERIMENTS.items()},
}
