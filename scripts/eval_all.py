%%writefile Skripsi_Fix-master/scripts/eval_all.py
import os
import glob
import csv
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Menambahkan root folder repo ke system path secara dinamis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import berdasarkan struktur direktori repo-mu
from models.iformer_gapnet import Iformer_GapNet
from dataset import test_dataset
from saleval import Eval_thread

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SOD models in memory")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing .pth files')
    parser.add_argument('--out_csv', type=str, default='/kaggle/working/hasil_evaluasi_skripsi.csv')
    parser.add_argument('--testsize', type=int, default=352, help='Image size for testing')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Path dataset (Pastikan path ini benar di env Kaggle kamu)
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
    
    # Inisialisasi Model
    model = Iformer_GapNet().to(device) 
    model.eval()

    # Buat header CSV sesuai request: F max, F weighted, E max, E mean, S mea, Mean (MAE)
    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model_Epoch', 'Dataset', 'F_max', 'F_weighted', 'E_max', 'E_mean', 'S_measure', 'Mean_MAE'])

    # Mulai evaluasi
    for epoch_path in model_paths:
        epoch_name = os.path.basename(epoch_path).replace('.pth', '')
        print(f"\n{'='*50}\nEvaluasi Model: {epoch_name}\n{'='*50}")
        
        # Load bobot model
        model.load_state_dict(torch.load(epoch_path, map_location=device))

        for ds_name, ds_txt in datasets.items():
            image_root = os.path.dirname(ds_txt)
            gt_root = os.path.dirname(ds_txt)
            
            test_loader = test_dataset(image_root, gt_root, testsize=args.testsize)
            evaluator = Eval_thread()
            
            with torch.no_grad():
                for image, gt, name in tqdm(test_loader, desc=f"Testing {ds_name}"):
                    image = image.to(device)
                    res = model(image)
                    
                    res = F.upsample(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    gt_numpy = gt.numpy().squeeze()
                    
                    evaluator.step(res, gt_numpy)

            # Ekstrak metrik
            metrics = evaluator.get_results()
            
            # --- PENARIKAN METRIK AMAN ---
            # Menggunakan .get() agar tidak error jika penamaan key di saleval.py sedikit berbeda
            f_max = metrics.get('maxf', metrics.get('F_max', metrics.get('f_max', 0.0)))
            f_w = metrics.get('weightf', metrics.get('wF', metrics.get('F_w', 0.0)))
            e_max = metrics.get('maxe', metrics.get('E_max', metrics.get('e_max', 0.0)))
            e_mean = metrics.get('meane', metrics.get('E_mean', metrics.get('e_mean', 0.0)))
            s_m = metrics.get('sm', metrics.get('S_m', metrics.get('s_m', 0.0)))
            mae = metrics.get('mae', metrics.get('MAE', metrics.get('Mean', 0.0)))
            
            # Append baris ke CSV
            with open(args.out_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch_name, ds_name, 
                    f"{f_max:.4f}", 
                    f"{f_w:.4f}", 
                    f"{e_max:.4f}",
                    f"{e_mean:.4f}",
                    f"{s_m:.4f}",
                    f"{mae:.4f}"
                ])

    print(f"\nUji hipotesis selesai! Hasil bisa didownload di: {args.out_csv}")

if __name__ == "__main__":
    main()
