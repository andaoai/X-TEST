"""
视觉容量实验 —— 测试 embedding 能否同时编码多个属性
"""
from experiments.vision.capacity.line_pos_rot import ExpLinePosRotCapacity
from experiments.vision.capacity.line_full import ExpLineFullCapacity

VISION_CAPACITY_EXPERIMENTS = {
    "vc_line_pos_rot": ExpLinePosRotCapacity,
    "vc_line_full": ExpLineFullCapacity,
}
