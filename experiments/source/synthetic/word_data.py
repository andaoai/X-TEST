"""
多字母单词数据集 v2 —— 每张图渲染 2-6 字母单词,每个字母独立颜色。

颜色分层自然产生 K=N_letters+1 个 token(spatial 模式 + 背景),让 self-attention
真正发挥作用。

数据构造:
  对每个目标单词(如 "apple" / "garden"):
    - 渲染 N 个变体:
      A) 正确单词 (valid):每个字母按字母序号分配不同颜色
      B) 干扰样本 (invalid),3 种均匀分布:
         - wrong_letter_1:替换 1 个字母为非词字母
         - wrong_letter_2:替换 2 个字母
         - scrambled:字母乱摆(独立随机 x,y 坐标,保证间距 ≥6 像素)

labels 结构:
  - word_match:    整体拼对/没拼对 (二分类)
  - word_id:       整词身份 (多分类,目标词数)
  - length:        词长 2/3/4/5/6 (5 分类)
  - mode:          valid/wrong1/wrong2/scrambled (4 分类,诊断用)
  - letter:        每张图每个 token 的字符 (per-token 多分类)
  - position:      字母在单词中的位次
  - color:         每 token 颜色名
  - letter_valid:  每 token 字符是否属于目标词对应位置
  - word:          字符串目标词

用法:
  ds = WordDatasetV2(target_words=DEFAULT_WORDS_50)
  ds.generate()
  rgb, labels = ds.rgbs(), ds.labels
"""
import os
import string
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from experiments.source.synthetic.config import (
    IMG_SIZE, EN_LETTERS, COLORS,
)


# 字符槽位:多字母在 64x64 图里水平排列的 x 中心
# 2 字母: 左 22, 右 42
# 3 字母: 左 18, 中 32, 右 46
# 4 字母: 左 16, 中 26, 中右 38, 右 48
# 5 字母: 10, 22, 32, 42, 54  (间距 10-12px,字宽 ~12px 不重叠)
# 6 字母: 8, 18, 27, 37, 46, 56
SLOT_X = {
    2: [22, 42],
    3: [18, 32, 46],
    4: [16, 26, 38, 48],
    5: [10, 22, 32, 42, 54],
    6: [8, 18, 27, 37, 46, 56],
}
CENTER_Y = 32  # 垂直居中

# 按词长选字号(平衡清晰度与不重叠)
# 注:4 字母词用 fs=20 会有少量粘连或碎裂,
# 模型 attention 池化对 K 变化是鲁棒的,故不追求 K 完美
FONT_SIZE_BY_LENGTH = {2: 26, 3: 22, 4: 20, 5: 16, 6: 12}

# 默认 50 词词表,按长度分层
DEFAULT_WORDS_50 = [
    # 2 字母 (8)
    "go", "no", "so", "do", "be", "me", "he", "it",
    # 3 字母 (12)
    "cat", "dog", "bat", "box", "sun", "map", "log", "run", "hat", "cup", "pen", "key",
    # 4 字母 (14)
    "time", "love", "hope", "mind", "word", "real", "good", "game", "fire", "blue",
    "dark", "gold", "tree", "open",
    # 5 字母 (10)
    "apple", "happy", "light", "music", "peace", "smart", "world", "heart", "water", "dream",
    # 6 字母 (6)
    "orange", "garden", "simple", "spring", "flower", "please",
]
assert len(DEFAULT_WORDS_50) == 50, f"词表数量异常: {len(DEFAULT_WORDS_50)}"

# 26 字母表(覆盖所有 50 词)
ALL_LETTERS = list(string.ascii_lowercase)


class WordDataset:
    """
    多字母单词数据集 v2:支持 2-6 字母,4 种生成模式,3 种 invalid 类型。
    """

    def __init__(self,
                 target_words: list[str] = None,
                 letters: list[str] = None,
                 colors: list[tuple] = None,
                 color_names: list[str] = None,
                 font_size: int = None,        # None = 按词长自适应
                 img_size: int = IMG_SIZE,
                 samples_per_word: int = 30,
                 invalid_ratio: float = 0.5,
                 seed: int = 42):
        self.target_words = target_words or DEFAULT_WORDS_50
        self.letters = letters or ALL_LETTERS
        self.colors = colors or list(COLORS.values())[:6]
        self.color_names = color_names or list(COLORS.keys())[:6]
        self.font_size = font_size   # None 表示自适应
        self.img_size = img_size
        self.samples_per_word = samples_per_word
        self.invalid_ratio = invalid_ratio
        self.seed = seed

        # 过滤掉不支持长度的词
        self.target_words = [w for w in self.target_words if len(w) in SLOT_X]
        if not self.target_words:
            raise ValueError("没有长度在 2-6 之间的目标词")

        self.samples: list = []
        self.labels: dict = {}
        self.token_metadata: list = []

    @staticmethod
    def _font(char: str, fs: int):
        if '一' <= char <= '鿿':
            for fp in [
                os.path.expanduser('~/.local/share/fonts/wqy-microhei.ttc'),
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
            ]:
                try:
                    return ImageFont.truetype(fp, fs)
                except OSError:
                    continue
        for fp in [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        ]:
            try:
                return ImageFont.truetype(fp, fs)
            except OSError:
                continue
        return ImageFont.load_default()

    def _render_one_letter(self, char, color_rgb, x_center, y_center=None, fs=None):
        """在 (x_center, y_center) 渲染一个字符,返回 RGB + mask。"""
        s = self.img_size
        if y_center is None:
            y_center = CENTER_Y
        if fs is None:
            fs = self.font_size
        img = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(img)
        f = self._font(char, fs)
        bb = d.textbbox((0, 0), char, font=f)
        cw = bb[2] - bb[0]
        ch = bb[3] - bb[1]
        x_pos = int(x_center - cw / 2 - bb[0])
        y_pos = int(y_center - ch / 2 - bb[1])
        d.text((x_pos, y_pos), char, fill=255, font=f)
        gray = np.array(img, dtype=np.float32) / 255.0
        rgb = np.zeros((s, s, 3), np.float32)
        for c in range(3):
            rgb[:, :, c] = gray * (color_rgb[c] / 255.0)
        mask = gray > 0.5
        return rgb, mask, cw  # 返回字宽,用于 scrambled 间距检查

    def _scrambled_positions(self, n: int, rng: random.Random, fs: int) -> list:
        """
        生成 n 个字母的独立随机位置,保证两两间距 ≥ MIN_GAP 像素。
        rejection sampling:不满足重抽(最多 200 次)。
        """
        s = self.img_size
        # 字框估计 ~ 14x18,留 6 像素间距 → 20 像素直径
        min_gap = max(6, fs * 0.5)
        positions = []
        max_attempts = 200
        for _ in range(n):
            for _ in range(max_attempts):
                x = rng.uniform(10, s - 10)
                y = rng.uniform(12, s - 12)
                ok = True
                for (px, py) in positions:
                    if abs(px - x) < min_gap and abs(py - y) < min_gap:
                        ok = False
                        break
                if ok:
                    positions.append((x, y))
                    break
            else:
                # 兜底:放在一个网格上
                grid_n = int(np.ceil(np.sqrt(n)))
                idx = len(positions)
                positions.append((10 + (s - 20) * (idx % grid_n) / max(1, grid_n - 1),
                                  12 + (s - 24) * (idx // grid_n) / max(1, grid_n - 1)))
        return positions

    def _render_word(self, chars: list[str], color_indices: list[int],
                     layout: str = "ordered", rng: random.Random = None,
                     fs: int = None):
        """
        渲染整张单词图。
        layout:
          - "ordered":按 SLOT_X 槽位水平排列
          - "scrambled":独立随机 x,y (但用 rng 决定)
        fs: 字号(默认按词长从 FONT_SIZE_BY_LENGTH 取)
        """
        n = len(chars)
        if n not in SLOT_X:
            raise ValueError(f"单词长度 {n} 不支持,目前 2-6")
        if fs is None:
            fs = self.font_size if self.font_size is not None else FONT_SIZE_BY_LENGTH[n]
        H, W = self.img_size, self.img_size
        rgb_full = np.zeros((H, W, 3), np.float32)
        masks = []
        cws = []

        if layout == "ordered":
            xs = SLOT_X[n]
            ys = [CENTER_Y] * n
        elif layout == "scrambled":
            if rng is None:
                rng = random.Random()
            xs_ys = self._scrambled_positions(n, rng, fs)
            xs = [p[0] for p in xs_ys]
            ys = [p[1] for p in xs_ys]
        else:
            raise ValueError(f"未知 layout: {layout}")

        for i, (ch, ci) in enumerate(zip(chars, color_indices)):
            cr = self.colors[ci]
            ch_rgb, ch_mask, cw = self._render_one_letter(ch, cr, xs[i], ys[i], fs=fs)
            rgb_full += ch_rgb
            masks.append(ch_mask)
            cws.append(cw)
        return rgb_full, masks

    def _make_invalid(self, word: str, mode: str, rng: random.Random) -> tuple:
        """
        返回 (chars_render, layout) 元组。
        mode: "wrong1" / "wrong2" / "scrambled"
        """
        chars = list(word)
        n = len(chars)
        if mode == "wrong1":
            pos = rng.randint(0, n - 1)
            while True:
                new_ch = rng.choice(self.letters)
                if new_ch != chars[pos]:
                    chars[pos] = new_ch
                    break
            return chars, "ordered"
        elif mode == "wrong2":
            positions = rng.sample(range(n), 2)
            for pos in positions:
                while True:
                    new_ch = rng.choice(self.letters)
                    if new_ch != chars[pos]:
                        chars[pos] = new_ch
                        break
            return chars, "ordered"
        elif mode == "scrambled":
            return chars, "scrambled"
        else:
            raise ValueError(f"未知 invalid mode: {mode}")

    def generate(self, verbose: bool = True):
        rng = random.Random(self.seed)
        self.samples = []
        self.labels = {}
        self.token_metadata = []

        # 3 种 invalid 模式均匀分布
        invalid_modes = ["wrong1", "wrong2", "scrambled"]

        for word in self.target_words:
            n = len(word)
            if n not in SLOT_X:
                continue
            for sample_idx in range(self.samples_per_word):
                # 为每个字母选一个颜色(允许重复)
                color_indices = [rng.randint(0, len(self.colors) - 1) for _ in range(n)]
                is_valid = rng.random() >= self.invalid_ratio

                if is_valid:
                    chars_render = list(word)
                    mode = "valid"
                    layout = "ordered"
                else:
                    mode = rng.choice(invalid_modes)
                    chars_render, layout = self._make_invalid(word, mode, rng)

                rgb, masks = self._render_word(chars_render, color_indices,
                                               layout=layout, rng=rng,
                                               fs=FONT_SIZE_BY_LENGTH[n])

                # per-token metadata(注意 chars_render 顺序 = 视觉顺序,
                # scrambled 时是 word 原序,不是坐标序——这是 ground truth 的选择)
                tokens = []
                for i, (ch, ci) in enumerate(zip(chars_render, color_indices)):
                    tokens.append({
                        "char": ch,
                        "position_in_word": i,
                        "color_idx": ci,
                        "color_name": self.color_names[ci],
                        "is_valid_in_target": (layout == "ordered" and ch == word[i]),
                        "mask": masks[i],
                    })

                sample = {
                    "rgb": rgb,
                    "target_word": word,
                    "rendered_word": "".join(chars_render),
                    "word_match": is_valid,
                    "mode": mode,             # "valid" / "wrong1" / "wrong2" / "scrambled"
                    "n_tokens": n,
                    "layout": layout,          # "ordered" / "scrambled"
                    "tokens": tokens,
                }
                idx = len(self.samples)
                self.samples.append(sample)
                self.token_metadata.append(tokens)

                # ── 写 labels ──
                for i, t in enumerate(tokens):
                    self._add_label("letter", f"{t['char']}_p{i}", idx)
                    self._add_label("letter_position", str(i), idx)
                    self._add_label("letter_color", t["color_name"], idx)
                    self._add_label("letter_valid", "yes" if t["is_valid_in_target"] else "no", idx)
                self._add_label("word_match", "yes" if is_valid else "no", idx)
                self._add_label("target_word", word, idx)
                self._add_label("n_tokens", str(n), idx)
                self._add_label("length", str(n), idx)
                self._add_label("word_id", word, idx)
                self._add_label("mode", mode, idx)

        if verbose:
            n_total = len(self.samples)
            n_valid = sum(1 for s in self.samples if s["word_match"])
            n_w1 = sum(1 for s in self.samples if s["mode"] == "wrong1")
            n_w2 = sum(1 for s in self.samples if s["mode"] == "wrong2")
            n_scr = sum(1 for s in self.samples if s["mode"] == "scrambled")
            print(f"  WordDataset v2: {n_total} 张, "
                  f"{len(self.target_words)} 目标词 × {self.samples_per_word} 变体")
            print(f"  valid={n_valid}  wrong1={n_w1}  wrong2={n_w2}  scrambled={n_scr}")
        return self

    def _add_label(self, field: str, value: str, idx: int):
        self.labels.setdefault(field, {}).setdefault(value, []).append(idx)

    def rgbs(self) -> np.ndarray:
        return np.stack([s["rgb"] for s in self.samples])

    def masks(self) -> np.ndarray:
        out = []
        for s in self.samples:
            m = np.zeros((self.img_size, self.img_size), np.float32)
            for t in s["tokens"]:
                m = np.maximum(m, t["mask"].astype(np.float32))
            out.append(m)
        return np.stack(out)


if __name__ == "__main__":
    ds = WordDataset(samples_per_word=6, invalid_ratio=0.5)
    ds.generate()
    print(f"\n样本数: {len(ds.samples)}")
    print(f"\nlabels 字段: {list(ds.labels.keys())}")
    for k, v in ds.labels.items():
        print(f"  {k}: {len(v)} 个 value, 总样本 {sum(len(idxs) for idxs in v.values())}")
    print(f"\nRGB shape: {ds.rgbs().shape}")
