import torch
import torch.nn as nn

pi = 3.141592653589793


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()

        # ===== Adaptive k CNN =====
        self.k_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # original parameter (kept for compatibility)
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))

        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3

        self.this_k = None  # will store k_map

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype

        # ===== Compute HSV components =====
        hue = torch.zeros(img.shape[0], img.shape[2], img.shape[3], device=device, dtype=dtypes)

        value = img.max(1)[0]
        img_min = img.min(1)[0]

        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value] % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        # ===== Adaptive k(x) =====
        k_map = self.k_net(img)                  # [B,1,H,W]
        k = 0.1 + 0.9 * k_map
        k = torch.clamp(k, 0.1, 1.0)

        # store for inverse
        self.this_k = k.detach()

        # ===== Illumination modulation =====
        I = value
        gamma = 1 + 0.5 * (1 - I)
        I_mod = torch.pow(I, gamma)
        I_mod = torch.clamp(I_mod, 0, 1)

        # ===== Modified intensity collapse =====
        color_sensitive = ((I_mod * 0.5 * pi).sin() + eps) ** k

        # ===== Construct HVI =====
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()

        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = I_mod  # IMPORTANT: use modulated intensity

        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8

        H = img[:, 0, :, :]
        V = img[:, 1, :, :]
        I = img[:, 2, :, :]

        # ===== clamp inputs =====
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I

        # ===== use average k for inverse =====
        if self.this_k is not None:
            k = torch.mean(self.this_k)
        else:
            k = self.density_k

        color_sensitive = ((v * 0.5 * pi).sin() + eps) ** k

        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)

        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1

        s = torch.sqrt(H**2 + V**2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi

        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)

        rgb = torch.cat([r, g, b], dim=1)

        if self.gated2:
            rgb = rgb * self.alpha

        return rgb