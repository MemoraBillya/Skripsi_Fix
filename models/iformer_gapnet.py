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
        
        # Di sini kita tidak melakukan proyeksi channel, 
        # kita biarkan GAPNet yang menyesuaikan di model.py.
        
    def forward(self, x):
        features = []

        # Ekstrak fitur Stride 2 agar jumlah fitur pas 5 buah.
        # Pada iFormer_t, downsample_layers[0] adalah nn.Sequential yang berisi:
        # [0] Conv2d_BN (Stride 2), [1] GELU, [2] EdgeResidual (Stride 2)

        x_s2 = self.backbone.downsample_layers[0][0](x)
        x_s2 = self.backbone.downsample_layers[0][1](x_s2)
        features.append(x_s2) # Fitur ke-1 (Stride 2): Output [B, 16, H/2, W/2]
        
        x_s4 = self.backbone.downsample_layers[0][2](x_s2)
        x_s4 = self.backbone.stages[0](x_s4)
        features.append(x_s4) # Fitur ke-2 (Stride 4): Output [B, 32, H/4, W/4]
        # -------------------------

        # Stage 2 (Stride 8)
        x_s8 = self.backbone.downsample_layers[1](x_s4)
        x_s8 = self.backbone.stages[1](x_s8)
        features.append(x_s8) # Fitur ke-3 (Stride 8)

        # Stage 3 (Stride 16)
        x_s16 = self.backbone.downsample_layers[2](x_s8)
        x_s16 = self.backbone.stages[2](x_s16)
        features.append(x_s16) # Fitur ke-4 (Stride 16)

        # Stage 4 (Stride 32)
        x_s32 = self.backbone.downsample_layers[3](x_s16)
        x_s32 = self.backbone.stages[3](x_s32)
        features.append(x_s32) # Fitur ke-5 (Stride 32)

        return features
        
        # iFormer-T architecture breakdown based on official implementation
        # Stage 1 (Stride 4)
        # x = self.backbone.downsample_layers[0](x)
        # x = self.backbone.stages[0](x)
        # features.append(x) # Output: [B, 32, H/4, W/4]

        # # Stage 2 (Stride 8)
        # x = self.backbone.downsample_layers[1](x)
        # x = self.backbone.stages[1](x)
        # features.append(x) # Output: [B, 64, H/8, W/8]

        # # Stage 3 (Stride 16)
        # x = self.backbone.downsample_layers[2](x)
        # x = self.backbone.stages[2](x)
        # features.append(x) # Output: [B, 128, H/16, W/16]

        # # Stage 4 (Stride 32)
        # x = self.backbone.downsample_layers[3](x)
        # x = self.backbone.stages[3](x)
        # features.append(x) # Output: [B, 256, H/32, W/32]

        # return features
