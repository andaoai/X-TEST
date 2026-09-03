"""
轮廓傅里叶描述子(Fourier descriptor)—— 形状的完整可逆表示。

把物体边界(轮廓)等弧长重采样成复数序列 z(t)=x(t)+i·y(t), 做 DFT 得 F[k]:
  · 位置 = F[0](质心, 复数)
  · 尺度 = |F[1]|(基频模长)
  · 旋转 = 所有系数的整体相位(乘 e^{iθ})
  · 形状 = 归一化复系数 F[k]/|F[1]|(保留相位!不是幅值)—— 平移/旋转/尺度不变
形状不变量用于判别;完整复系数(含 F0 与相位)用于重建:
  保留 k=-K..K 阶 → 逆 DFT 得带限轮廓点 → 填充多边形 → mask。
K→M/2 时 Parseval 保证无损;K 越小越压缩,细节按频率光滑截断。
轮廓按弧长参数化(不按径向角), 故十字/非星形/带洞(多环)形状都能表示。
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw

from algorithms.base import BaseAlgorithm


def _contours(mask):
    m = (mask > 0).astype(np.uint8) * 255
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    rings = []
    if hier is not None and len(cnts):
        hier = hier[0]
        # (轮廓点, 是否外环)
        for c, h in zip(cnts, hier):
            if len(c) >= 3:
                rings.append((c[:, 0, :].astype(np.float64), h[3] < 0))  # h[3]==-1 外环
    return rings


def _resample(pts, M):
    pts = np.asarray(pts, np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1] + 1e-9
    t = np.linspace(0, total, M, endpoint=False)
    x = np.interp(t, cum, closed[:, 0])
    y = np.interp(t, cum, closed[:, 1])
    return x + 1j * y


def encode_fourier(mask, K=16, M=512):
    """返回 (coeffs_list, meta)。coeffs_list: 每个环一个 (F0, Fpos(2K,), Fneg(2K,)) 复数。"""
    rings = _contours(mask)
    out = []
    for pts, is_outer in rings:
        z = _resample(pts, M)
        F = np.fft.fft(z) / M
        Fp = np.array([F[k] for k in range(1, K + 1)], np.complex128)
        Fn = np.array([F[M - k] for k in range(1, K + 1)], np.complex128)
        out.append((F[0], Fp, Fn, is_outer))
    return out, dict(K=K, M=M)


def decode_fourier(coeffs, meta, out_shape):
    K, M = meta["K"], meta["M"]
    img = Image.new("L", (out_shape[1], out_shape[0]), 0)
    draw = ImageDraw.Draw(img)
    # 先画所有外环(白), 再画所有内环/洞(黑)挖空
    ordered = sorted(coeffs, key=lambda c: 0 if c[3] else 1)
    for F0, Fp, Fn, is_outer in ordered:
        F = np.zeros(M, np.complex128)
        F[0] = F0
        for k in range(1, K + 1):
            F[k] = Fp[k - 1]
            F[M - k] = Fn[k - 1]
        z = np.fft.ifft(F) * M
        pts = list(zip(z.real, z.imag))
        draw.polygon(pts, fill=255 if is_outer else 0)
    return (np.asarray(img) > 0).astype(np.uint8)


def coeff_count(coeffs):
    """存储的实数个数: 每环 F0(2) + 正负频率各 K 个复数(各2实数)= 2+4K。"""
    K = len(coeffs[0][1])
    return sum(2 + 4 * K for _ in coeffs)


def _signed_area(z):
    return 0.5 * np.sum((z * np.roll(np.conj(z), -1)).imag)


def canonical_invariants(mask, K=16, M=512):
    """轮廓对齐到标准姿态后取复傅里叶系数(保留相位), 平移/旋转/尺度/起点不变。

    步骤: 质心置零 → 旋转使基频 F1 为实正数(消旋转)→ 逆时针化(消镜像/顺逆)
          → 循环移位到"最右点"为起点(消起点相位)→ F[k]/|F1|。
    同一形状任意位姿 → 同一向量;不同形状 → 不同向量。
    """
    rings = _contours(mask)
    pts, _ = max(rings, key=lambda r: len(r[0]))
    z = _resample(pts, M)
    z -= z.mean()
    F = np.fft.fft(z) / M
    z = z * np.exp(-1j * np.angle(F[1]))           # 旋转: F1 → 实正
    if _signed_area(z) < 0:                          # 统一逆时针
        z = z[::-1].copy()
    start = int(np.argmax(z.real))                   # 最右点为标准起点
    z = np.roll(z, -start)
    F = np.fft.fft(z) / M
    scale = np.abs(F[1]) + 1e-9
    Fp = np.array([F[k] / scale for k in range(1, K + 1)])
    inv = np.concatenate([Fp.real, Fp.imag])
    return inv.astype(np.float32)


def pose_vector(coeffs, H, W):
    """位姿: 质心位置(2) + 尺度(1) + 基频相位=朝向(1)。"""
    F0, Fp, Fn, _ = coeffs[0]
    return np.array([F0.real / W, F0.imag / H, np.abs(Fp[0]) / W,
                     np.angle(Fp[0]) / np.pi], np.float32)


POSE_K = (2, 3)        # 位姿朝向用角谐波 F2/F3 的复数相位(随旋转旋相)
A_REF_FRAC = 0.02      # 参考面积 = 画面的 2%(用于把大小维居中)


def _pose_features(mask):
    """7 维位姿: 质心(2) + F2/F3 复数相位(4, 朝向) + 居中 log 面积(1, 大小)。"""
    m = mask > 0
    ys, xs = np.where(m)
    H, W = mask.shape
    area = max(len(xs), 1)
    cx, cy = xs.mean(), ys.mean()
    dx = (xs - cx).astype(np.float64)
    dy = (ys - cy).astype(np.float64)
    r = np.sqrt(dx * dx + dy * dy)
    phi = np.arctan2(dy, dx)
    rw = r / (r.mean() + 1e-6)                      # 半径加权, 突出外圈角度结构
    pose = [cx / W - 0.5, cy / H - 0.5]
    for k in POSE_K:
        Fk = (rw * np.exp(1j * k * phi)).sum() / rw.sum()
        pose.extend([Fk.real, Fk.imag])
    pose.append(np.log(area / (A_REF_FRAC * H * W)))   # 居中: 中等≈0, 大→+, 小→-
    return np.asarray(pose, np.float32)


class FourierShapeAlgo(BaseAlgorithm):
    """轮廓傅里叶描述子作为 embedding。

    内容(平移/旋转/尺度不变): 归一化傅里叶模长 |F_k|/能量 —— 比径向统计完整,
        且同系数可逆 DFT 还原形状。
    位姿(弱权重): 质心 F0(位置)、二阶矩主轴(朝向)、系数能量(大小)。
    输出自然长度扁平向量(K 内容 + 5 位姿),L2 归一化。
    """
    name = "fourier_shape"
    uses_rgb = False

    # 该算法可逆: reconstruct() 用高 K 完整复系数逆 DFT 还原 mask。
    reversible = True

    def __init__(self, K: int = 12, pos_weight: float = 0.70,
                 ori_weight: float = 0.50, size_weight: float = 1.0,
                 recon_K: int = 64):
        self.K = K
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.size_weight = size_weight
        self.recon_K = recon_K

    def reconstruct(self, inputs):
        """完整复系数(高 K)逆 DFT 重建 mask → 用于第 7 指标"重建还原率"。"""
        inputs = np.asarray(inputs)
        out = []
        for i in range(inputs.shape[0]):
            m = inputs[i]
            coeffs, meta = encode_fourier(m, K=self.recon_K)
            out.append(decode_fourier(coeffs, meta, m.shape))
        return np.stack(out)

    def _features(self, mask):
        # 内容: 轮廓傅里叶模长(低阶鲁棒), 归一化 → 平移/旋转/尺度不变的形状谱
        coeffs, _ = encode_fourier(mask, K=self.K)
        F0, Fp, Fn, _ = coeffs[0]
        energy = np.sqrt((np.abs(Fp) ** 2).sum() + (np.abs(Fn) ** 2).sum()) + 1e-9
        content = (np.abs(Fp) / energy).astype(np.float32)
        # 位姿(自包含): 质心 + F2/F3 朝向 + 居中 log 面积
        pose = _pose_features(mask)
        return content, pose

    def encode(self, inputs, verbose=True):
        inputs = np.asarray(inputs)
        out = []
        for i in range(inputs.shape[0]):
            c, p = self._features(inputs[i])
            c = c / (np.linalg.norm(c) + 1e-8)
            p = p / (np.linalg.norm(p) + 1e-8)
            pw = p.copy()
            pw[0:2] *= self.pos_weight   # 质心位置
            pw[2:6] *= self.ori_weight   # F2/F3 朝向
            pw[6]   *= self.size_weight   # log 面积(大小), 傅里叶模长尺度归一化后靠此维
            v = np.concatenate([c, pw])
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append(v.astype(np.float32))
        return np.stack(out, axis=0)
