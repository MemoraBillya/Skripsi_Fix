import torch
import torch.nn as nn
from models.iformer import iFormer_t 

class iFormerGapNet(nn.Module):
    def __init__(self, pretrained=True, args_path=None):
        super(iFormerGapNet, self).__init__()
        self.backbone = iFormer_t(pretrained=pretrained) #Pretrained
        
        # Hapus classification head
        if hasattr(self.backbone, 'head'):
            del self.backbone.head
        if hasattr(self.backbone, 'norm'):
            del self.backbone.norm
            
        # Channel asli iFormer-Tiny: [32, 64, 128, 256]
        # GAPNet default MobileNet: [16, 24, 32, 96, 160] (tapi decoder hanya pakai 24 ke atas)
        # Kita proyeksikan agar channelnya lebih "bersahabat" dengan decoder GAPNet yang ringan
        # atau biarkan GAPNet menyesuaikan channelnya via enc_channels.
        
        # Di sini kita tidak melakukan proyeksi channel, 
        # kita biarkan GAPNet yang menyesuaikan di model.py.
        
    def forward(self, x):
        features = []
        
        # iFormer-T architecture breakdown based on official implementation
        # Stage 1 (Stride 4)
        x = self.backbone.downsample_layers[0](x)
        x = self.backbone.stages[0](x)
        features.append(x) # Output: [B, 32, H/4, W/4]

        # Stage 2 (Stride 8)
        x = self.backbone.downsample_layers[1](x)
        x = self.backbone.stages[1](x)
        features.append(x) # Output: [B, 64, H/8, W/8]

        # Stage 3 (Stride 16)
        x = self.backbone.downsample_layers[2](x)
        x = self.backbone.stages[2](x)
        features.append(x) # Output: [B, 128, H/16, W/16]

        # Stage 4 (Stride 32)
        x = self.backbone.downsample_layers[3](x)
        x = self.backbone.stages[3](x)
        features.append(x) # Output: [B, 256, H/32, W/32]

        return features
