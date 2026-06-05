"""文字实验"""
from experiments.text.lang import ExpLangSeparation
from experiments.text.position import ExpPositionInvariance
from experiments.text.color import ExpColorSeparation
from experiments.text.pos_encode import ExpPositionEncoding
from experiments.text.char import ExpCharSeparation

TEXT_EXPERIMENTS = {
    "exp1": ExpLangSeparation(),
    "exp2": ExpPositionInvariance(),
    "exp3": ExpColorSeparation(),
    "exp4": ExpPositionEncoding(),
    "exp5": ExpCharSeparation(),
}
