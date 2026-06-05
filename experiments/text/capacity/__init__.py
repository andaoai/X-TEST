"""
容量实验 —— 测试 embedding 能否同时编码多个属性
"""
from experiments.text.capacity.char_pos import ExpCharPosCapacity
from experiments.text.capacity.char_scale import ExpCharScaleCapacity
from experiments.text.capacity.full import ExpFullCapacity

CAPACITY_EXPERIMENTS = {
    "c_char_pos": ExpCharPosCapacity,
    "c_char_scale": ExpCharScaleCapacity,
    "c_full": ExpFullCapacity,
}
