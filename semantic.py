import torch
import torch.nn as nn
import torchvision.models as models

class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.conv = nn.Conv2d(512, 1, 1)   # 🔥 1-channel attention

    def forward(self, x):
        feat = self.encoder(x)
        attn = self.conv(feat)

        attn = torch.sigmoid(
            torch.nn.functional.interpolate(
                attn, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        )

        return attn