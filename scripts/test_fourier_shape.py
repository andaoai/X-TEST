"""
轮廓傅里叶描述子: 用多少阶谐波能还原多少? 与 radial/无损sprite 对照。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.fourier_shape import encode_fourier, decode_fourier, coeff_count
from experiments.source.synthetic.shape import generate_shape_dataset

for _fp in [os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc")]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False
S = 640

def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / ((a | b).sum() + 1e-8)

masks, labels = generate_shape_dataset()
def pick(shape, size="medium", rot=0, pos=(320,320)):
    s = set(labels["shape"][shape]) & set(labels["size"][size]) \
        & set(labels["rotation"][str(rot)]) & set(labels["position"][str(pos)])
    return sorted(s)[len(sorted(s))//2]
samples = [("圆", masks[pick("circle")]),
           ("三角", masks[pick("triangle", rot=45, pos=(210,430))]),
           ("正方形", masks[pick("square")]),
           ("十字", masks[pick("cross", rot=135)]),
           ("线", masks[pick("line","large",90)]),
           ("椭圆", masks[pick("ellipse","large",45,(210,210))])]
def blob_pts(seed, r, rot):
    rng=np.random.RandomState(seed); nv=rng.randint(9,15)
    ang=np.sort(rng.uniform(0,2*np.pi,nv)); rad=r*rng.uniform(0.55,1.45,nv)
    e2=np.random.RandomState(seed+7).uniform(0.45,0.9)
    th=np.deg2rad(rot); c,s=np.cos(th),np.sin(th)
    x,y=rad*np.cos(ang)*e2,rad*np.sin(ang); return list(zip(x*c-y*s,x*s+y*c))
for b in range(3):
    img=Image.new("L",(S,S),0); ImageDraw.Draw(img).polygon([(320+x,320+y) for x,y in blob_pts(7000+b,60,b*40)],fill=1)
    samples.append((f"随机块{b}", np.asarray(img,np.uint8)))

# ── 扫 K ──
Ks = [2, 4, 8, 16, 32, 64, 128]
print(f"{'K阶':>4}{'实数个数':>9}" + "".join(f"{n:>8}" for n,_ in samples))
curve = {}
for K in Ks:
    scores = []
    for name, m in samples:
        co, meta = encode_fourier(m, K=K)
        rec = decode_fourier(co, meta, m.shape)
        scores.append(iou(m, rec))
    n_num = coeff_count(encode_fourier(samples[0][1], K=K)[0])
    curve[K] = scores
    print(f"{K:>4}{n_num:>9}" + "".join(f"{v:>8.3f}" for v in scores))

# ── 重建图(K=32)──
Kshow = 32
fig, axes = plt.subplots(3, len(samples), figsize=(22, 9))
for col, (name, m) in enumerate(samples):
    co, meta = encode_fourier(m, K=Kshow)
    rec = decode_fourier(co, meta, m.shape)
    axes[0, col].imshow(m[::3,::3], cmap="gray_r"); axes[0, col].set_title(f"{name} 原图", fontsize=11)
    axes[1, col].imshow(rec[::3,::3], cmap="gray_r"); axes[1, col].set_title(f"傅里叶 K={Kshow}\nIoU={iou(m,rec):.3f} ({coeff_count(co)}个数)", fontsize=10)
    ov = np.zeros((S,S,3),np.uint8)
    ov[...,1][(m>0)&(rec>0)]=255; ov[...,0][(m>0)&(rec==0)]=255; ov[...,2][(m==0)&(rec>0)]=255
    axes[2, col].imshow(ov[::3,::3]); axes[2, col].set_title("差异(绿对/红丢/蓝多)", fontsize=10)
    for r in range(3):
        axes[r,col].set_xticks([]); axes[r,col].set_yticks([])
fig.suptitle("轮廓傅里叶描述子: 边界→复数 DFT(保留相位)→逆 DFT 重建。阶数越多越逼近, 位置=F0/相位=旋转/模长=尺度", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
out="results/shape_study/19_fourier_descriptor.png"
fig.savefig(out, dpi=105); print("→", out)
