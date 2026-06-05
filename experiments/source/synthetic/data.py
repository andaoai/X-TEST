"""
合成数据生成 —— 彩色字符 → mask + RGB 图像。

用法:
  from experiments.synthetic.data import Dataset
  ds = Dataset(letters=["A","B"], colors=["Red","Blue"])
  ds.generate()
  masks = ds.masks()   # (N, H, W)  二值
  rgbs  = ds.rgbs()    # (N, H, W, 3) 彩色
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from experiments.source.synthetic.config import (IMG_SIZE, FONT_SIZE, EN_LETTERS,
    ZH_CHARS, COLORS, POSITIONS)


class Dataset:
    def __init__(self, letters=None, chinese=None, colors=None, positions=None,
                 img_size=IMG_SIZE, font_size=FONT_SIZE):
        self.letters   = letters   or EN_LETTERS
        self.chinese   = chinese   or ZH_CHARS
        self.colors    = dict(colors) if colors else dict(COLORS)
        self.positions = dict(positions) if positions else dict(POSITIONS)
        self.img_size  = img_size
        self.font_size = font_size
        self.samples: list = []
        self.labels: dict  = {}

    # ── 内部 ──
    @staticmethod
    def _font(char, fs):
        if '一' <= char <= '鿿':
            for fp in ['C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/msyh.ttc',
                       'C:/Windows/Fonts/simsun.ttc']:
                try: return ImageFont.truetype(fp, fs)
                except OSError: continue
        try: return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fs)
        except OSError: return ImageFont.load_default()

    def _render(self, char, color_rgb, pos):
        s = self.img_size
        img = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(img)
        f = self._font(char, self.font_size)
        bb = d.textbbox((0, 0), char, font=f)
        d.text((pos[0]-bb[0], pos[1]-bb[1]), char, fill=255, font=f)
        gray = np.array(img, dtype=np.float32) / 255.0
        rgb = np.zeros((s, s, 3), np.float32)
        for c in range(3):
            rgb[:,:,c] = gray * (color_rgb[c] / 255.0)
        return rgb, gray > 0.5

    # ── 生成 ──
    def generate(self, verbose=True):
        self.samples, self.labels = [], {}
        items = ([(c, "EN") for c in self.letters] +
                 [(c, "ZH") for c in self.chinese])
        for char, lang in items:
            for cn, cr in self.colors.items():
                for pn, px in self.positions.items():
                    rgb, m = self._render(char, cr, px)
                    s = dict(rgb=rgb, mask=m, label=char, lang=lang,
                             color=cn, position=pn, pos_xy=px)
                    self.samples.append(s)
                    for field in ["lang","color","position","label"]:
                        self.labels.setdefault(field,{}).setdefault(
                            s[field],[]).append(len(self.samples)-1)
        if verbose:
            print(f"  数据: {len(self.samples)} 张 "
                  f"({len(self.letters)}英+{len(self.chinese)}中 "
                  f"x{len(self.colors)}色 x{len(self.positions)}位)")
        return self

    def masks(self):
        return np.stack([s["mask"].astype(np.float32) for s in self.samples])

    def rgbs(self):
        return np.stack([s["rgb"] for s in self.samples])
