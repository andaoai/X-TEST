"""
视觉可分性实验 —— 测试属性是否被编码在 embedding 中
"""
from experiments.vision.separability.pixel_position import ExpPixelPositionSeparation
from experiments.vision.separability.pixel_count import ExpPixelCountSeparation
from experiments.vision.separability.line_position import ExpLinePositionSeparation
from experiments.vision.separability.line_rotation import ExpLineRotationSeparation
from experiments.vision.separability.line_length import ExpLineLengthSeparation

VISION_SEPARABILITY_EXPERIMENTS = {
    "vs_pixel_pos": ExpPixelPositionSeparation,
    "vs_pixel_cnt": ExpPixelCountSeparation,
    "vs_line_pos": ExpLinePositionSeparation,
    "vs_line_rot": ExpLineRotationSeparation,
    "vs_line_len": ExpLineLengthSeparation,
}
