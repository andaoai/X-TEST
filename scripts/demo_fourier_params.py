"""
演示: 一套傅里叶复系数 = 位置(F0) + 大小(|F1|) + 旋转(整体相位) + 形状(归一化复系数)。
只改其中一个因子, 逆 DFT 重建, 验证各管各的、且可任意组合/还原。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.fourier_shape import encode_fourier, decode_fourier, _resample, _contours
from experiments.source.synthetic.shape import generate_shape_dataset

for _fp in [os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc")]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False
S = 640
K = 48

def iou(a,b):
    a,b=a>0,b>0; return (a&b).sum()/((a|b).sum()+1e-8)

masks, labels = generate_shape_dataset()
def pick(shape,size="medium",rot=0,pos=(320,320)):
    s=set(labels["shape"][shape])&set(labels["size"][size])&set(labels["rotation"][str(rot)])&set(labels["position"][str(pos)])
    return sorted(s)[len(s)//2]

tri = masks[pick("triangle")]
sq  = masks[pick("square")]

co_t, meta = encode_fourier(tri, K=K)
F0t, Fpt, Fnt, _ = co_t[0]
co_s, _ = encode_fourier(sq, K=K)
F0s, Fps, Fns, _ = co_s[0]

def rebuild(F0, Fp, Fn):
    return decode_fourier([(F0, Fp, Fn, True)], meta, tri.shape)

# 原始
orig = rebuild(F0t, Fpt, Fnt)
# ① 只改位置 F0 → 平移(其它不动)
mov = rebuild(F0t + complex(-160, -120), Fpt, Fnt)
# ② 只改大小: 非零系数 ×1.5(绕质心放大)
big = rebuild(F0t, Fpt*1.5, Fnt*1.5)
# ③ 只改旋转: 非零系数 ×e^{iθ}(绕质心转 90°)
th = np.deg2rad(90)
rot = rebuild(F0t, Fpt*np.exp(1j*th), Fnt*np.exp(1j*th))
# ④ 只换形状: 正方形的形状系数, 重标定到三角形的"大小"(非零系数能量)和位置
def energy(Fp, Fn): return np.sqrt((np.abs(Fp)**2).sum() + (np.abs(Fn)**2).sum())
E_t, E_s = energy(Fpt, Fnt), energy(Fps, Fns)
swap = rebuild(F0t, Fps/E_s*E_t, Fns/E_s*E_t)
# 全阶(K=256)完整重建
full, _ = encode_fourier(tri, K=256)
full_rec = decode_fourier(full, dict(K=256, M=512), tri.shape)

panels = [("原始三角\n(一套复系数)", orig),
          ("只改 F0 → 平移\n位置=F0", mov),
          ("非零系数×1.5 → 放大\n大小=非零系数能量", big),
          ("非零系数×e^{i90°} → 旋转\n旋转=整体相位", rot),
          ("换正方形形状系数\n→ 正方,位姿不变", swap)]
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax,(title,img) in zip(axes, panels):
    ax.imshow(img[::3,::3], cmap="gray_r"); ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("一套傅里叶复系数同时编码 位置/大小/旋转/形状: 只动一个因子,逆DFT重建只改对应属性", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.92])
out="results/shape_study/20_fourier_params.png"; fig.savefig(out,dpi=110)
print("→", out)

E_t = np.sqrt((np.abs(Fpt)**2).sum()+(np.abs(Fnt)**2).sum())
print(f"参数读数(三角): 位置 F0=({F0t.real:.0f},{F0t.imag:.0f})  大小(非零系数能量)={E_t:.1f}px  旋转=整体相位因子")
print(f"全阶 K=256 完整重建 IoU = {iou(tri, full_rec):.4f}  (K→M/2 时 DFT 可逆, 轮廓无损)")
print(f"改位置后 IoU vs 原图 = {iou(tri, mov):.3f}(形状相同、位置不同→应<1); 放大后 = {iou(tri,big):.3f}")
