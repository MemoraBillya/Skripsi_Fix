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
import py_sod_metrics as M

# =========================================================================
# FUNGSI LOSS (Identik dengan Training agar Val Loss Akurat)
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

class CEOLoss(nn.Module):
    def __init__(self, criterion=BCEDiceLoss, ignore_index=False, supervision=8):
        super(CEOLoss, self).__init__()
        self.criterion = criterion
        self.ignore_index = ignore_index
        self.supervision = supervision

    def forward(self, inputs, targets):
        criterion = self.criterion
        losses = []
        for i in [0, 1, 2, 5]:
            losses.append(criterion(inputs[:, i, :, :], targets[:, i, :, :], ignore_index=False))
        for i in range(3, 5):
            losses.insert(i, criterion(inputs[:, i, :, :], targets[:, i, :, :], ignore_index=self.ignore_index))

        if self.supervision == 8:
            losses[3] = criterion(inputs[:, 3, :, :], targets[:, 2, :, :], ignore_index=False)
            loss_overall = losses[:1] + losses[3:5]
        else:
            loss_overall = losses[:1] + losses[3:5]
        return sum(loss_overall)/len(loss_overall)*3

# =========================================================================
# FUNGSI EVALUASI
# =========================================================================
@torch.no_grad()
def evaluate_dataset(model, dataloader, criterion=None, calc_loss=False, device='cuda'):
    model.eval()
    
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE_metric = M.MAE()
    
    epoch_loss = []
    bar = tqdm(dataloader, leave=False)
    
    for iter, (input_tensor, target_tensor) in enumerate(bar):
        input_var = input_tensor.to(device)
        target_var = target_tensor.to(device).float()
        
        output = model(input_var)
        
        # 1. Hitung Loss (HANYA JIKA DIMINTA - KHUSUS VAL SET)
        if calc_loss and criterion is not None:
            # Perhitungan loss granular menggunakan semua level mask
            loss = criterion(output, target_var)
            epoch_loss.append(loss.item())
            bar.set_description(f"Loss: {sum(epoch_loss) / len(epoch_loss):.4f}")
        
        # 2. Hitung Metrik (Hanya butuh mask level 0)
        target_squeezed = target_var[:, 0, :, :].squeeze(1) if len(target_var.shape) == 4 else target_var.squeeze(1)
        preds = (output[:, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        gts = (target_squeezed.cpu().numpy() * 255).astype(np.uint8)
        
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis=0)
            gts = np.expand_dims(gts, axis=0)
            
        for i in range(preds.shape[0]):
            FM.step(preds[i], gts[i])
            WFM.step(preds[i], gts[i])
            SM.step(preds[i], gts[i])
            EM.step(preds[i], gts[i])
            MAE_metric.step(preds[i], gts[i])
            
    # Rekap Hasil
    avg_loss = sum(epoch_loss) / len(epoch_loss) if calc_loss else 0.0
    
    fm_res = FM.get_results()['fm']
    em_res = EM.get_results()['em']
    
    res = {
        'loss': avg_loss,
        'F_max': fm_res['curve'].max(),
        'F_w': WFM.get_results()['wfm'],
        'S_m': SM.get_results()['sm'],
        'E_max': em_res['curve'].max(),
        'E_mean': em_res['curve'].mean(),
        'MAE': MAE_metric.get_results()['mae']
    }
    return res

# =========================================================================
# EKSEKUSI UTAMA
# =========================================================================
def main():
    # --- KONFIGURASI PATH ---
    data_dir = '/kaggle/working/data/'
    
    # Folder tempat model 1-30 Anda berada (Bisa dibaca)
    model_dir = '/kaggle/input/datasets/sejutakerinduan/mdl-img-bs16lr1-7e-3' 
    
    # Folder tempat menyimpan hasil evaluasi (Harus di /working/ agar bisa ditulis)
    output_dir = '/kaggle/working/' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.GAPNet(arch='iformer_tiny', pretrained=False).to(device)
    
    # Jika model dilatih pakai DataParallel, bungkus modelnya juga di sini
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criteria = CEOLoss(criterion=BCEDiceLoss, supervision=8)
    
    # Dataset Transforms
    NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)), 
                        np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))]
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(384, 384),
        myTransforms.ToTensor()
    ])

    # Pastikan file partisi val disalin ke folder data
    partisi_val_path = '/kaggle/input/datasets/billydawson/partisi-train/DUTS-TR_val_20.txt'
    if os.path.exists(partisi_val_path):
        shutil.copy(partisi_val_path, os.path.join(data_dir, 'DUTS-TR-VAL.lst'))
    
    # --- INISIALISASI DATALOADER ---
    # 1. Loader Validasi (process_label=True untuk hitung Loss Granular)
    valLoader = torch.utils.data.DataLoader(
        Dataset(data_dir, 'DUTS-TR-VAL', transform=valDataset, process_label=True),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True) # BATCH KECIL & WORKER 0 (SUPER AMAN)

    # 2. Loader Benchmark Test (process_label=False karena tidak hitung Loss)
    test_names = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]
    testLoaders = {}
    for t_name in test_names:
        testLoaders[t_name] = torch.utils.data.DataLoader(
            Dataset(data_dir, t_name, transform=valDataset, process_label=False),
            batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Buka file log untuk mencatat hasil eval
    log_eval = open(os.path.join(output_dir, 'evaluation_results.txt'), 'w')

    print("Memulai Evaluasi Offline (Epoch 1 - 30)...\n")
    
    # Loop Evaluasi setiap Epoch
    for epoch in range(1, 31):
        # Ambil model dari model_dir
        model_path = os.path.join(model_dir, f'model_{epoch}.pth')
        
        if not os.path.exists(model_path):
            print(f"Model epoch {epoch} tidak ditemukan di {model_path}, melewati...")
            continue
            
        # Load bobot
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n=======================================================")
        print(f"🚀 Evaluasi Model Epoch {epoch}")
        
        # --- 1. Evaluasi Validation Set ---
        print(">> Dataset Validasi (DUTS-TR Val 20%) -> Menghitung Loss & Metrik...")
        val_res = evaluate_dataset(model, valLoader, criteria, calc_loss=True, device=device)
        
        # PERBAIKAN: Cetak SEMUA metrik untuk Validation
        val_log = f"Epoch {epoch} [VAL] -> Loss: {val_res['loss']:.4f} | F-max: {val_res['F_max']:.4f} | F-w: {val_res['F_w']:.4f} | S-m: {val_res['S_m']:.4f} | E-max: {val_res['E_max']:.4f} | E-mean: {val_res['E_mean']:.4f} | MAE: {val_res['MAE']:.4f}"
        print(val_log)
        log_eval.write(val_log + "\n")
        
        # --- 2. Evaluasi Test Sets ---
        for t_name in test_names:
            print(f">> Benchmark: {t_name} -> Menghitung Metrik...")
            t_res = evaluate_dataset(model, testLoaders[t_name], criterion=None, calc_loss=False, device=device)
            
            # PERBAIKAN: Cetak SEMUA metrik untuk Test Sets (tanpa Loss)
            t_log = f"          [{t_name}] -> F-max: {t_res['F_max']:.4f} | F-w: {t_res['F_w']:.4f} | S-m: {t_res['S_m']:.4f} | E-max: {t_res['E_max']:.4f} | E-mean: {t_res['E_mean']:.4f} | MAE: {t_res['MAE']:.4f}"
            print(t_log)
            log_eval.write(t_log + "\n")
            
    log_eval.close()
    print("\n✅ Seluruh proses evaluasi offline selesai! Hasil tersimpan di evaluation_results.txt")

if __name__ == '__main__':
    main()
