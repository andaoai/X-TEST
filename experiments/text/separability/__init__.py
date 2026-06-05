"""
可分性实验 —— 测试属性是否被编码在 embedding 中
"""
from experiments.text.separability.character import ExpCharSeparation
from experiments.text.separability.position import ExpPositionSeparation
from experiments.text.separability.rotation import ExpRotationSeparation
from experiments.text.separability.scale import ExpScaleSeparation

SEPARABILITY_EXPERIMENTS = {
    "s_char": ExpCharSeparation,
    "s_position": ExpPositionSeparation,
    "s_rotation": ExpRotationSeparation,
    "s_scale": ExpScaleSeparation,
}
