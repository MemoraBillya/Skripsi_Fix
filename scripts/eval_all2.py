import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from tqdm import tqdm
from models import model as net
from dataset import Dataset
import transforms as myTransforms

# =========================================================================
# FUNGSI LOSS (Hanya Butuh BCEDiceLoss untuk Mask Akhir)
# =========================================================================
def BCEDiceLoss(inputs, targets, ignore_index=False):
    bce = CrossEntropyLoss(inputs, targets, ignore_index)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

def CrossEntropyLoss(inputs, targets, ignore_index=False):
    index = [i for i in range(targets.size()[0]) if not torch.all(targets[i] == 10)]
    targets = targets[index, :, :]
    inputs = inputs[index, :, :]

    if ignore_index:
        valid_mask = (targets != 255)
        targets_valid = targets.clone()
        targets_valid[targets == 255] = 0
        BCE_func = nn.BCELoss(reduction='none')
        bce = BCE_func(inputs, targets_valid)
        bce = bce * valid_mask
        bce = torch.sum(bce) / (torch.sum(valid_mask) + 1e-8)  
    else:
        bce = F.binary_cross_entropy(inputs, targets)
    return bce

# =========================================================================
# FUNGSI EVALUASI (SUPER RINGAN - HANYA LOSS MASK AKHIR)
# =========================================================================
@torch.no_grad()
def evaluate_loss_only(model, dataloader, criterion, device='cuda'):
    model.eval()
    epoch_loss = []
    bar = tqdm(dataloader, leave=False, desc="Calculating Val Loss")
    
    for input_tensor, target_tensor in bar:
        input_var = input_tensor.to(device)
        
        # Karena process_label=False, target hanya 1 lapis
        target_var = target_tensor.to(device).float()
        target_squeezed = target_var.squeeze(1)
        
        output = model(input_var)
        
        # Hitung Loss HANYA pada tebakan utama (output[:, 0, :, :])
        loss = criterion(output[:, 0, :, :], target_squeezed)
        epoch_loss.append(loss.item())
        
        if len(epoch_loss) % 10 == 0:
            bar.set_description(f"Val Loss: {sum(epoch_loss) / len(epoch_loss):.4f}")
            
    return sum(epoch_loss) / len(epoch_loss)

# =========================================================================
# EKSEKUSI UTAMA
# =========================================================================
def main():
    # --- KONFIGURASI PATH ---
    data_dir = '/kaggle/working/data/'
    
    # Folder tempat model 1-30 Anda berada
    model_dir = '/kaggle/input/datasets/sejutakerinduan/mdl-img-bs8lr1-7e-4' 
    output_dir = '/kaggle/working/' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.GAPNet(arch='iformer_tiny', pretrained=False).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Gunakan fungsi dasar BCEDiceLoss secara langsung (tanpa CEOLoss)
    criterion = BCEDiceLoss
    
    NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)), 
                        np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))]
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(384, 384),
        myTransforms.ToTensor()
    ])

    partisi_val_path = '/kaggle/input/datasets/billydawson/partisi-train/DUTS-TR_val_20.txt'
    if os.path.exists(partisi_val_path):
        shutil.copy(partisi_val_path, os.path.join(data_dir, 'DUTS-TR-VAL.lst'))
    
    # --- INISIALISASI DATALOADER ---
    # process_label=False membuat CPU tidak perlu menghitung 6 lapis mask!
    valLoader = torch.utils.data.DataLoader(
        Dataset(data_dir, 'DUTS-TR-VAL', transform=valDataset, process_label=False),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    log_eval = open(os.path.join(output_dir, 'val_loss_only_results.txt'), 'w')

    print("Memulai Evaluasi Cepat (Hanya Val Loss Level 0)...\n")
    
    for epoch in range(1, 31):
        model_path = os.path.join(model_dir, f'model_{epoch}.pth')
        
        if not os.path.exists(model_path):
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Eksekusi kalkulasi loss
        val_loss = evaluate_loss_only(model, valLoader, criterion, device=device)
        
        # Cetak dan catat hasilnya
        val_log = f"Epoch {epoch:02d} | Val Loss (Mask Akhir): {val_loss:.4f}"
        print(val_log)
        log_eval.write(val_log + "\n")
            
    log_eval.close()
    print("\n✅ Evaluasi cepat selesai! Hasil tersimpan di val_loss_only_results.txt")

if __name__ == '__main__':
    main()
