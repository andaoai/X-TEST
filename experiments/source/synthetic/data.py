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
    ZH_CHARS, COLORS, POSITIONS, ROTATIONS, FONT_SIZES)


class Dataset:
    def __init__(self, letters=None, chinese=None, colors=None, positions=None,
                 rotations=None, font_sizes=None, img_size=IMG_SIZE, font_size=FONT_SIZE):
        self.letters     = letters     or EN_LETTERS
        self.chinese     = chinese     or ZH_CHARS
        self.colors      = dict(colors) if colors else dict(COLORS)
        self.positions   = dict(positions) if positions else dict(POSITIONS)
        self.rotations   = dict(rotations) if rotations else dict(ROTATIONS)
        self.font_sizes  = dict(font_sizes) if font_sizes else dict(FONT_SIZES)
        self.img_size    = img_size
        self.font_size   = font_size
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

    def _render(self, char, color_rgb, pos, angle=0, font_size=None):
        s = self.img_size
        fs = font_size or self.font_size
        # 先在大画布上渲染，旋转后裁剪，避免旋转时出界
        pad = int(s * 0.5)
        big = s + pad * 2
        img = Image.new("L", (big, big), 0)
        d = ImageDraw.Draw(img)
        f = self._font(char, fs)
        bb = d.textbbox((0, 0), char, font=f)
        # 字符居中绘制在大画布上
        cx, cy = big // 2, big // 2
        d.text((cx - bb[0] - (bb[2] - bb[0]) // 2,
                cy - bb[1] - (bb[3] - bb[1]) // 2), char, fill=255, font=f)
        if angle != 0:
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
        # 裁剪回原尺寸，按 pos 偏移
        left = pad + (pos[0] - s // 2)
        top  = pad + (pos[1] - s // 2)
        img = img.crop((left, top, left + s, top + s))
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
                    for rn, rv in self.rotations.items():
                        for sn, sv in self.font_sizes.items():
                            rgb, m = self._render(char, cr, px, rv, sv)
                            s = dict(rgb=rgb, mask=m, label=char, lang=lang,
                                     color=cn, position=pn, pos_xy=px,
                                     rotation=rn, size=sn)
                            self.samples.append(s)
                            for field in ["lang","color","position","label","rotation","size"]:
                                self.labels.setdefault(field,{}).setdefault(
                                    s[field],[]).append(len(self.samples)-1)
        if verbose:
            print(f"  数据: {len(self.samples)} 张 "
                  f"({len(self.letters)}英+{len(self.chinese)}中 "
                  f"x{len(self.colors)}色 x{len(self.positions)}位"
                  f"x{len(self.rotations)}转 x{len(self.font_sizes)}大小)")
        return self

    def masks(self):
        return np.stack([s["mask"].astype(np.float32) for s in self.samples])

    def rgbs(self):
        return np.stack([s["rgb"] for s in self.samples])
