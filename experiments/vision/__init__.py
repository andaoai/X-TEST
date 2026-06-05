"""视觉实验 —— 控制变量实验"""
from experiments.vision.independence import VISION_INDEPENDENCE_EXPERIMENTS
from experiments.vision.separability import VISION_SEPARABILITY_EXPERIMENTS
from experiments.vision.capacity import VISION_CAPACITY_EXPERIMENTS

VISION_EXPERIMENTS = {
    # 独立性实验
    **{k: v() for k, v in VISION_INDEPENDENCE_EXPERIMENTS.items()},

    # 可分性实验
    **{k: v() for k, v in VISION_SEPARABILITY_EXPERIMENTS.items()},

    # 容量实验
    **{k: v() for k, v in VISION_CAPACITY_EXPERIMENTS.items()},
}
