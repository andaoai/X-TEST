"""
独立性实验 —— 测试属性变化是否影响 embedding
"""
from experiments.text.independence.position import ExpPositionIndependence
from experiments.text.independence.rotation import ExpRotationIndependence
from experiments.text.independence.scale import ExpScaleIndependence

INDEPENDENCE_EXPERIMENTS = {
    "i_position": ExpPositionIndependence,
    "i_rotation": ExpRotationIndependence,
    "i_scale": ExpScaleIndependence,
}
