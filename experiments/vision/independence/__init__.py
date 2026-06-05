"""
视觉独立性实验 —— 测试属性变化是否影响 embedding
"""
from experiments.vision.independence.pixel_position import ExpPixelPositionIndependence
from experiments.vision.independence.line_position import ExpLinePositionIndependence
from experiments.vision.independence.line_rotation import ExpLineRotationIndependence
from experiments.vision.independence.line_length import ExpLineLengthIndependence

VISION_INDEPENDENCE_EXPERIMENTS = {
    "vi_pixel_pos": ExpPixelPositionIndependence,
    "vi_line_pos": ExpLinePositionIndependence,
    "vi_line_rot": ExpLineRotationIndependence,
    "vi_line_len": ExpLineLengthIndependence,
}
