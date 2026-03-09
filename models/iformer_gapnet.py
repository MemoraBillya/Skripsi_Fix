import torch
import torch.nn as nn
from models.iformer import iFormer_t 

class iFormerGapNet(nn.Module):
    def __init__(self, pretrained=True, args_path=None):
        super(iFormerGapNet, self).__init__()
        # self.backbone = iFormer_t(pretrained=pretrained) #Pretrained
        self.backbone = iFormer_t(pretrained=False)

        # Load custom pretrained weights jika True
        if pretrained:
            pretrained_path = '/kaggle/input/iformer-skripsi/pytorch/default/1/iFormer_t.pth'
            print(f"Loading iFormer-T pretrained weights from: {pretrained_path}")
            
            # Load file .pth
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Handle jika weight dibungkus dalam key 'model' (umum terjadi pada model timm/DeiT)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # Load state dict ke backbone
            self.backbone.load_state_dict(state_dict, strict=False)
        
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
