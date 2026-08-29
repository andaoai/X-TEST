"""
测试 algorithms/image_decomposition.py 分解效果。

生成 4 类测试图,每类运行 token 化,输出:
  1) 原图 + 标注 token 索引号的可视化
  2) 每张图的 token 统计(K / K_fg / K_bg / 像素数)

输出:
  results/decomposition_test/grid.png    ← 4 类图对比
  results/decomposition_test/detail_*.png ← 每张原图大图
  results/decomposition_test/stats.txt    ← 文字统计

用法:
  uv run python scripts/test_decomposition.py
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# PIL 字体(支持中文,用于 header)—— cv2.putText 不支持 Unicode
_PIL_FONT_PATH = None
for _fp in [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]:
    if os.path.exists(_fp):
        _PIL_FONT_PATH = _fp
        break
_PIL_FONT = None


def _draw_text_pil(canvas_bgr: np.ndarray, text: str,
                   x: int, y: int, color_bgr=(255, 255, 255), fs: int = 14):
    """PIL 写文字(cv2 不支持中文),画到 BGR 画布。"""
    global _PIL_FONT
    h, w = canvas_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    if _PIL_FONT is None:
        try:
            _PIL_FONT = ImageFont.truetype(_PIL_FONT_PATH, fs)
        except Exception:
            _PIL_FONT = ImageFont.load_default()
    d.text((x, y), text, fill=(color_bgr[2], color_bgr[1], color_bgr[0]),
           font=_PIL_FONT)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.image_decomposition import (
    decompose_image_to_tokens,
    visualize_tokens,
    token_stats,
)

OUT_DIR = "results/decomposition_test"
os.makedirs(OUT_DIR, exist_ok=True)


# ── 工具:在画布上写英文单词 ──
def _font(fs: int):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, fs)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(img: Image.Image, text: str, x: int, y: int,
               rgb: tuple, fs: int = 14):
    d = ImageDraw.Draw(img)
    d.text((x, y), text, fill=rgb, font=_font(fs))


# ── 4 类测试图生成器 ──
def make_single_word() -> np.ndarray:
    """场景 1: 单个单词 'go' (2 字母, 不同颜色, 标准有序位置)"""
    h = w = 64
    rgb = np.zeros((h, w, 3), np.float32)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    _draw_text(img, "g", 14, 20, (255, 50, 50), fs=24)   # 红
    _draw_text(img, "o", 34, 20, (50, 50, 255), fs=24)   # 蓝
    return np.asarray(img, np.float32) / 255.0


def make_multi_word() -> np.ndarray:
    """场景 2: 多词 'cat' + 'dog' (两个词分别在不同 y 条带)"""
    h = w = 64
    rgb = np.zeros((h, w, 3), np.float32)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    _draw_text(img, "c", 10, 10,  (255, 50, 50), fs=18)
    _draw_text(img, "a", 24, 10,  (50, 255, 50), fs=18)
    _draw_text(img, "t", 38, 10,  (50, 50, 255), fs=18)
    _draw_text(img, "d", 10, 36,  (255, 255, 50), fs=18)
    _draw_text(img, "o", 24, 36,  (255, 50, 255), fs=18)
    _draw_text(img, "g", 38, 36,  (50, 255, 255), fs=18)
    return np.asarray(img, np.float32) / 255.0


def make_same_color_split() -> np.ndarray:
    """场景 3: 同色不相邻 (两个 'o' 用同色, 应被连通域拆成 2 个 token)"""
    h = w = 64
    rgb = np.zeros((h, w, 3), np.float32)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    _draw_text(img, "o", 10, 20, (255, 0, 0), fs=24)   # 左红 o
    _draw_text(img, "o", 38, 20, (255, 0, 0), fs=24)   # 右红 o(同色, 应拆)
    return np.asarray(img, np.float32) / 255.0


def make_color_proximity() -> np.ndarray:
    """场景 4: 字母粘连 ('ab' 同色, 挨近) — 抗粘连测试"""
    h = w = 64
    rgb = np.zeros((h, w, 3), np.float32)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    _draw_text(img, "a", 8, 20,  (255, 200, 50), fs=22)   # 黄
    _draw_text(img, "b", 28, 20, (255, 200, 50), fs=22)   # 黄(挨近, 同色 — 可能合并)
    _draw_text(img, "c", 48, 20, (50, 255, 200), fs=22)   # 青(隔离)
    return np.asarray(img, np.float32) / 255.0


# ── 主流程 ──
def main():
    scenarios = [
        ("01_single_word", "单词 'go' (2 字母, 异色, 有序)",
         make_single_word(), ["g", "o"]),
        ("02_multi_word",  "多词 'cat'+'dog' (6 字母, 异色, 上下条带)",
         make_multi_word(), ["c", "a", "t", "d", "o", "g"]),
        ("03_same_color",  "同色不相邻 (2 个红 o, 应被连通域拆成 2 token)",
         make_same_color_split(), ["o1", "o2"]),
        ("04_proximity",   "字母粘连 (2 黄 ab 挨近 + 1 青 c 隔离)",
         make_color_proximity(), ["a", "b", "c"]),
    ]

    all_stats = []

    # ── 每张图单独大图 ──
    for tag, desc, rgb, hints in scenarios:
        tokens, bg_mask = decompose_image_to_tokens(rgb, mode="spatial",
                                                    char_hints=hints)
        stats = token_stats(tokens)
        stats["tag"] = tag
        stats["desc"] = desc
        all_stats.append(stats)

        # 可视化
        viz = visualize_tokens(rgb, tokens, show_index=True)
        # 在图上方加文字描述
        # ── 1) 彩色叠加图(原图 + 索引号) ──
        viz = visualize_tokens(rgb, tokens, show_index=True)
        out_path = f"{OUT_DIR}/{tag}.png"
        header = np.full((24, viz.shape[1], 3), 30, dtype=np.uint8)
        header = _draw_text_pil(header,
                                f"{tag}  K={stats['K']}(BG={stats['K_bg']}+FG={stats['K_fg']})",
                                5, 5, color_bgr=(0, 255, 255), fs=13)
        big = np.vstack([header, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)])
        cv2.imwrite(out_path, big)
        print(f"  → {out_path}  K={stats['K']} (BG={stats['K_bg']} + FG={stats['K_fg']})")

        # ── 2) 每个 token 一张 B/W mask,横向并排 ──
        K = len(tokens)
        masks_bw = []
        for j, (_, m) in enumerate(tokens):
            mask_u8 = (m > 0.5).astype(np.uint8) * 255
            masks_bw.append(mask_u8)
        row = np.hstack(masks_bw)
        row_3ch = np.repeat(row[:, :, None], 3, axis=2)   # (H, W) → (H, W, 3)
        cell_w = row_3ch.shape[1] // K
        header_canvas = np.full((20, row_3ch.shape[1], 3), 30, dtype=np.uint8)
        for j, (hsv_id, m) in enumerate(tokens):
            label = f"j={j} px={int(m.sum())} h=({hsv_id[0]},{hsv_id[1]},{hsv_id[2]})"
            sub = header_canvas[:, j*cell_w:(j+1)*cell_w].copy()
            sub = _draw_text_pil(sub, label, 4, 2, color_bgr=(0, 255, 0), fs=11)
            header_canvas[:, j*cell_w:(j+1)*cell_w] = sub
        masks_combined = np.vstack([header_canvas, row_3ch])
        masks_path = f"{OUT_DIR}/{tag}_masks.png"
        cv2.imwrite(masks_path, masks_combined)
        print(f"  → {masks_path}  ({K} 张 B/W mask)")

    # ── 4 张图合成 grid ──
    images = []
    for tag, desc, rgb, hints in scenarios:
        tokens, _ = decompose_image_to_tokens(rgb, mode="spatial",
                                              char_hints=hints)
        viz = visualize_tokens(rgb, tokens, show_index=True)
        header = np.full((24, viz.shape[1], 3), 30, dtype=np.uint8)
        header = _draw_text_pil(header, tag.split("_", 1)[1],
                                5, 5, color_bgr=(0, 255, 255), fs=13)
        images.append(np.vstack([header, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)]))
    # 2x2 grid
    top = np.hstack(images[:2])
    bot = np.hstack(images[2:])
    grid = np.vstack([top, bot])
    grid_path = f"{OUT_DIR}/grid.png"
    cv2.imwrite(grid_path, grid)
    print(f"\n  → {grid_path}  (2x2 概览)")

    # ── 文字统计 ──
    with open(f"{OUT_DIR}/stats.txt", "w") as f:
        f.write(f"{'tag':<20} {'K':>3} {'BG':>3} {'FG':>3} "
                f"{'min':>5} {'max':>5} {'mean':>6}  desc\n")
        for s in all_stats:
            f.write(f"{s['tag']:<20} {s['K']:>3} {s['K_bg']:>3} {s['K_fg']:>3} "
                    f"{s['min_size']:>5} {s['max_size']:>5} "
                    f"{s['mean_size']:>6.0f}  {s['desc']}\n")
    print(f"  → {OUT_DIR}/stats.txt\n")

    # 打印到终端
    with open(f"{OUT_DIR}/stats.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
