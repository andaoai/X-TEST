"""文字实验 —— 控制变量实验"""
from experiments.text.independence import INDEPENDENCE_EXPERIMENTS
from experiments.text.separability import SEPARABILITY_EXPERIMENTS
from experiments.text.capacity import CAPACITY_EXPERIMENTS

TEXT_EXPERIMENTS = {
    # 独立性实验
    **{k: v() for k, v in INDEPENDENCE_EXPERIMENTS.items()},

    # 可分性实验
    **{k: v() for k, v in SEPARABILITY_EXPERIMENTS.items()},

    # 容量实验
    **{k: v() for k, v in CAPACITY_EXPERIMENTS.items()},
}
