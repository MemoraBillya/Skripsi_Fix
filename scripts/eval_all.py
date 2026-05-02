import os
import sys
import glob
import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import py_sod_metrics

# Tambahkan root direktori ke system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================================================================
# PENTING: UBAH BARIS INI SESUAI NAMA CLASS ASLI MODELMU
# Ganti "Iformer_GapNet" dengan nama class yang ada di iformer_gapnet.py
# ======================================================================
from models.iformer_gapnet import iFormerGapNet
from dataset import Dataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SOD with PySODMetrics")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default='/kaggle/working/hasil_pysod_skripsi.csv')
    parser.add_argument('--testsize', type=int, default=352)
    return parser.parse_args()

def main():
    args = get_args()
    
    datasets = {
        "PASCAL-S": "/kaggle/input/sod-skripsi/PASCAL-S.txt",
        "HKU-IS": "/kaggle/input/sod-skripsi/HKU-IS.txt",
        "ECSSD": "/kaggle/input/sod-skripsi/ECSSD.txt",
        "DUTS-TE": "/kaggle/input/sod-skripsi/DUTS-TE.txt",
        "DUT-OMRON": "/kaggle/input/sod-skripsi/DUT-OMRON.txt",
        "DUTS-TR_val_20": "/kaggle/input/datasets/billydawson/partisi-train/DUTS-TR_val_20.txt"
    }

    model_paths = sorted(glob.glob(os.path.join(args.model_dir, "*.pth")))
    if not model_paths:
        print(f"Error: Tidak ada file .pth di {args.model_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")
    
    # ======================================================================
    # PENTING: UBAH JUGA DI SINI (Samakan dengan nama Class yang di-import)
    # ======================================================================
    model = iFormerGapNet().to(device) 
    model.eval()

    # Buat header CSV
    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model_Epoch', 'Dataset', 'F_max', 'F_weighted', 'E_max', 'E_mean', 'S_measure', 'Mean_MAE'])

    for epoch_path in model_paths:
        epoch_name = os.path.basename(epoch_path).replace('.pth', '')
        print(f"\n{'='*50}\nEvaluasi Model: {epoch_name}\n{'='*50}")
        
        model.load_state_dict(torch.load(epoch_path, map_location=device))

        for ds_name, ds_txt in datasets.items():
            image_root = os.path.dirname(ds_txt)
            gt_root = os.path.dirname(ds_txt)
            
            test_loader = Dataset(image_root, gt_root, testsize=args.testsize)
            
            # Inisialisasi Instance PySODMetrics
            FM = py_sod_metrics.Fmeasure()
            WFM = py_sod_metrics.WeightedFmeasure()
            SM = py_sod_metrics.Smeasure()
            EM = py_sod_metrics.Emeasure()
            MAE = py_sod_metrics.MAE()
            
            with torch.no_grad():
                for image, gt, name in tqdm(test_loader, desc=f"Testing {ds_name}"):
                    image = image.to(device)
                    res = model(image)
                    
                    res = F.upsample(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    gt_numpy = gt.numpy().squeeze()
                    
                    # Konversi array ke format yang dimengerti PySODMetrics
                    # Prediksi diubah ke rentang 0-255 (float)
                    pred_np = (res * 255).astype(np.float32)
                    
                    # Ground Truth diubah ke binary mask murni (0 atau 255, uint8)
                    gt_np = (gt_numpy * 255).astype(np.uint8)
                    gt_np = np.where(gt_np > 128, 255, 0).astype(np.uint8)
                    
                    # Beri makan metrik satu per satu
                    FM.step(pred_np, gt_np)
                    WFM.step(pred_np, gt_np)
                    SM.step(pred_np, gt_np)
                    EM.step(pred_np, gt_np)
                    MAE.step(pred_np, gt_np)

            # Ekstrak hasil dari masing-masing objek PySODMetrics
            fm_res = FM.get_results()['fm']
            wfm_res = WFM.get_results()['wfm']
            sm_res = SM.get_results()['sm']
            em_res = EM.get_results()['em']
            mae_res = MAE.get_results()['mae']
            
            # Pengambilan nilai yang spesifik
            f_max = fm_res['curve'].max()
            f_w = wfm_res
            e_max = em_res['curve'].max()
            e_mean = em_res['curve'].mean()
            s_m = sm_res
            mae_m = mae_res
            
            with open(args.out_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch_name, ds_name, 
                    f"{f_max:.4f}", 
                    f"{f_w:.4f}", 
                    f"{e_max:.4f}",
                    f"{e_mean:.4f}",
                    f"{s_m:.4f}",
                    f"{mae_m:.4f}"
                ])

    print(f"\nUji hipotesis selesai! Hasil bisa didownload di: {args.out_csv}")

if __name__ == "__main__":
    main()
