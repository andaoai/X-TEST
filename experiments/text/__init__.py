"""文字实验 —— key = 文件名, 不会冲突"""
from experiments.text.lang import ExpLangSeparation
from experiments.text.position import ExpPositionInvariance
from experiments.text.color import ExpColorSeparation
from experiments.text.pos_encode import ExpPositionEncoding
from experiments.text.char import ExpCharSeparation
from experiments.text.rotation_sep import ExpRotationSeparation
from experiments.text.rotation_inv import ExpRotationInvariance

TEXT_EXPERIMENTS = {
    "lang":         ExpLangSeparation(),
    "position":     ExpPositionInvariance(),
    "color":        ExpColorSeparation(),
    "pos_encode":   ExpPositionEncoding(),
    "char":         ExpCharSeparation(),
    "rotation_sep": ExpRotationSeparation(),
    "rotation_inv": ExpRotationInvariance(),
}
