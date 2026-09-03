"""
视觉可分性实验 —— 测试属性是否被编码在 embedding 中
"""
from experiments.vision.separability.pixel_position import ExpPixelPositionSeparation
from experiments.vision.separability.pixel_count import ExpPixelCountSeparation
from experiments.vision.separability.line_position import ExpLinePositionSeparation
from experiments.vision.separability.line_rotation import ExpLineRotationSeparation
from experiments.vision.separability.line_length import ExpLineLengthSeparation
from experiments.vision.separability.shape_position import ExpShapePositionSeparation
from experiments.vision.separability.shape_type import ExpShapeTypeSeparation
from experiments.vision.separability.shape_size import ExpShapeSizeSeparation
from experiments.vision.separability.shape_rotation import ExpShapeRotationSeparation

VISION_SEPARABILITY_EXPERIMENTS = {
    "vs_pixel_pos": ExpPixelPositionSeparation,
    "vs_pixel_cnt": ExpPixelCountSeparation,
    "vs_line_pos": ExpLinePositionSeparation,
    "vs_line_rot": ExpLineRotationSeparation,
    "vs_line_len": ExpLineLengthSeparation,
    # 形状 × 位置编码实验(640×640)
    "vs_shape_pos": ExpShapePositionSeparation,
    "vs_shape_type": ExpShapeTypeSeparation,
    "vs_shape_size": ExpShapeSizeSeparation,
    "vs_shape_rot": ExpShapeRotationSeparation,
}
