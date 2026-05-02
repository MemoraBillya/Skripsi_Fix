import os
import glob
import csv
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Menambahkan root folder repo ke system path agar bisa import dari folder models/ dan dataset.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sesuaikan dengan nama fungsi/class di repo kamu
from models.iformer_gapnet import Iformer_GapNet
from dataset import test_dataset
from saleval import Eval_thread

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate all epochs in memory without saving images")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing .pth files')
    parser.add_argument('--out_csv', type=str, default='evaluation_results.csv', help='Path to save the output CSV')
    parser.add_argument('--testsize', type=int, default=352, help='Image size for testing')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Kumpulan path dataset di Kaggle
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
        print(f"Error: Tidak ada file .pth ditemukan di {args.model_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Inisialisasi model
    # Jika iformer butuh parameter, masukkan di sini (misal: backbone='iformer-t')
    model = Iformer_GapNet().to(device) 
    model.eval()

    # Buat header CSV
    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model_Epoch', 'Dataset', 'MAE', 'F_beta', 'S_m', 'E_m'])

    for epoch_path in model_paths:
        epoch_name = os.path.basename(epoch_path).replace('.pth', '')
        print(f"\n{'='*40}\nEvaluating: {epoch_name}\n{'='*40}")
        
        # Load bobot
        model.load_state_dict(torch.load(epoch_path, map_location=device))

        for ds_name, ds_txt in datasets.items():
            image_root = os.path.dirname(ds_txt)
            gt_root = os.path.dirname(ds_txt)
            
            # Load dataloader
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

            # Hitung rata-rata
            metrics = evaluator.get_results()
            
            # Append ke CSV
            with open(args.out_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch_name, ds_name, 
                    f"{metrics['mae']:.4f}", 
                    f"{metrics['f_beta']:.4f}", 
                    f"{metrics['s_m']:.4f}",
                    f"{metrics['e_m']:.4f}"
                ])

    print(f"\nSelesai! Hasil tersimpan di: {args.out_csv}")

if __name__ == "__main__":
    main()
