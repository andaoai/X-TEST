"""
合成数据配置 —— 数据从哪里来: 字母/颜色/位置的定义。

这是实验的"数据源"部分。
"""
from pathlib import Path
from algorithms.base import EMBEDDING_DIM

# 输出
OUTPUT_ROOT = Path(__file__).parent.parent.parent.parent / "results"

# 图像参数
IMG_SIZE   = 64
FONT_SIZE  = 28
SEED       = 42

# ── 数据组合定义 ──
EN_LETTERS   = ["A","B","C","D","E","F","G","H","X","Z"]
ZH_CHARS     = ["我","你","他","天","地","人","山","水","火","风"]

COLORS: dict[str, tuple] = {
    "Red":(255,0,0), "Green":(0,255,0), "Blue":(0,0,255),
    "Yellow":(255,255,0), "Cyan":(0,255,255), "Magenta":(255,0,255),
    "White":(255,255,255), "Orange":(255,128,0),
}

POSITIONS: dict[str, tuple] = {
    "TL":(8,8), "TR":(48,8), "CT":(28,28), "BL":(8,48), "BR":(48,48),
}

ROTATIONS: dict[str, int] = {
    "0": 0, "90": 90, "180": 180, "270": 270,
}

FONT_SIZES: dict[str, int] = {
    "small": 18, "medium": 28, "large": 38,
}

# 色板
COLOR_HEX = dict(zip(COLORS.keys(),
    ["#e74c3c","#2ecc71","#3498db","#f1c40f","#1abc9c","#9b59b6","#ecf0f1","#e67e22"]))
POS_HEX   = dict(zip(POSITIONS.keys(),
    ["#e74c3c","#3498db","#2ecc71","#9b59b6","#e67e22"]))
ROTATION_HEX = dict(zip(ROTATIONS.keys(),
    ["#3498db","#e74c3c","#2ecc71","#e67e22"]))
