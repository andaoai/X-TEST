"""
Gabor 提升 —— 固定核, 零训练, 三通道分离。FFT 批量卷积。
所有 Gabor 私有参数在这里, 不污染 config.py。
"""
import numpy as np
from algorithms.base import BaseAlgorithm, EMBEDDING_DIM


class GaborLift(BaseAlgorithm):
    name = "gabor_lift"
    uses_rgb = True

    n_orient = 12
    n_scales = 3
    sigmas   = [2.5, 4.0, 6.0]
    omegas   = [0.3, 0.2, 0.15]

    def _build(self, h, w):
        if hasattr(self, "_kffts"):
            return
        thetas = np.linspace(0, np.pi, self.n_orient + 1)[:-1]
        kffts = []
        for sigma, omega in zip(self.sigmas, self.omegas):
            ks = max(int(sigma * 6) | 1, 7)
            for th in thetas:
                half = ks // 2
                y, x = np.mgrid[-half:half + 1, -half:half + 1]
                xt = x * np.cos(th) + y * np.sin(th)
                yt = -x * np.sin(th) + y * np.cos(th)
                gauss = np.exp(-(xt**2 / (2*sigma**2) + yt**2 / (2*(sigma*0.5)**2)))
                k = gauss * np.cos(omega * xt)
                k -= k.mean()
                k /= np.linalg.norm(k) + 1e-8
                # pad & shift for FFT
                kp = np.zeros((h, w), dtype=np.float32)
                sh, sw = (h-ks)//2, (w-ks)//2
                kp[sh:sh+ks, sw:sw+ks] = k
                kffts.append(np.fft.rfft2(np.fft.ifftshift(kp)))
        self._kffts = np.stack(kffts)
        self._nf    = len(kffts)

    @staticmethod
    def _channels(rgb):
        R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
        L  = 0.299*R + 0.587*G + 0.114*B
        RG = R - G
        BY = B - (R+G)/2.0
        return np.stack([L, RG, BY], axis=1)

    def encode(self, inputs, verbose=True):
        N, H, W = inputs.shape[:3]
        self._build(H, W)
        nf = self._nf
        chs = self._channels(inputs)

        if verbose:
            print(f"  [Gabor] {nf}核 x 3通道, {N}张 (FFT)")

        raw_dim = 3 * self.n_scales * self.n_orient
        raw = np.zeros((N, raw_dim), dtype=np.float32)
        col = 0

        for ci in range(3):
            for n in range(N):
                img_f = np.fft.rfft2(chs[n, ci])
                for fi in range(nf):
                    resp = np.fft.irfft2(img_f * self._kffts[fi], s=(H, W))
                    raw[n, col + fi] = (np.maximum(0, resp)**2).sum()
            col += nf

        # per-(scale, channel) normalize
        for s in range(self.n_scales):
            for ci in range(3):
                start = ci*nf + s*self.n_orient
                block = raw[:, start:start + self.n_orient]
                raw[:, start:start + self.n_orient] = block / (
                    np.linalg.norm(block, axis=1, keepdims=True) + 1e-8)

        # 确保输出 EMBEDDING_DIM 维
        if raw_dim < EMBEDDING_DIM:
            emb = np.pad(raw, ((0,0),(0,EMBEDDING_DIM - raw_dim)))
        elif raw_dim > EMBEDDING_DIM:
            X = raw - raw.mean(axis=0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            emb = X @ Vt[:EMBEDDING_DIM].T
        else:
            emb = raw

        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
