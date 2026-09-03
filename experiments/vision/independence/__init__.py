"""
视觉独立性实验 —— 测试属性变化是否影响 embedding
"""
from experiments.vision.independence.pixel_position import ExpPixelPositionIndependence
from experiments.vision.independence.line_position import ExpLinePositionIndependence
from experiments.vision.independence.line_rotation import ExpLineRotationIndependence
from experiments.vision.independence.line_length import ExpLineLengthIndependence
from experiments.vision.independence.shape_position import ExpShapePositionIndependence
from experiments.vision.independence.shape_rotation import ExpShapeRotationIndependence

VISION_INDEPENDENCE_EXPERIMENTS = {
    "vi_pixel_pos": ExpPixelPositionIndependence,
    "vi_line_pos": ExpLinePositionIndependence,
    "vi_line_rot": ExpLineRotationIndependence,
    "vi_line_len": ExpLineLengthIndependence,
    # 形状 × 位置编码实验(640×640): 位置编码预期失败(对照可分性)
    "vi_shape_pos": ExpShapePositionIndependence,
    "vi_shape_rot": ExpShapeRotationIndependence,
}
