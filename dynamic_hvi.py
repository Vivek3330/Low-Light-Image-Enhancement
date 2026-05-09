import torch
import torch.nn as nn

class DynamicHVI(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.delta_h = nn.Conv2d(32, 1, 1)
        self.scale_s = nn.Conv2d(32, 1, 1)
        self.scale_i = nn.Conv2d(32, 1, 1)
        self.scale_c = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]

        max_rgb, _ = torch.max(x, dim=1, keepdim=True)
        min_rgb, _ = torch.min(x, dim=1, keepdim=True)
        delta = max_rgb - min_rgb + 1e-6

        # Original HSV parts
        I = max_rgb
        S = delta / (max_rgb + 1e-6)
        h = (g - b) / delta   # simplified hue

        # Learnable adjustments
        feat = self.encoder(x)

        delta_h = torch.tanh(self.delta_h(feat))
        scale_s = torch.sigmoid(self.scale_s(feat))
        scale_i = torch.sigmoid(self.scale_i(feat))
        scale_c = torch.sigmoid(self.scale_c(feat))

        # Apply dynamic mapping
        h_new = h + delta_h
        S_new = S * scale_s
        I_new = I * scale_i

        # Polarization
        H = torch.cos(torch.pi * h_new)
        V = torch.sin(torch.pi * h_new)

        # Adaptive collapse
        C = scale_c * torch.sin(torch.pi * I_new / 2)

        # Final
        H_hat = C * S_new * H
        V_hat = C * S_new * V

        return torch.cat([H_hat, V_hat, I_new], dim=1)