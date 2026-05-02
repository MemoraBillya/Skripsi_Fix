import os
import sys
import glob
import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import py_sod_metrics as M

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import GAPNet 

def get_args():
    parser = argparse.ArgumentParser(description="Evaluasi SOD Lengkap")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default='/kaggle/working/hasil_evaluasi_skripsi.csv')
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=384)
    return parser.parse_args()

def main():
    args = get_args()
    datasets = {
        "PASCAL-S": "/kaggle/input/sod-skripsi/PASCAL-S.txt",
        "HKU-IS": "/kaggle/input/sod-skripsi/HKU-IS.txt",
        "ECSSD": "/kaggle/input/sod-skripsi/ECSSD.txt",
        "DUTS-TE": "/kaggle/input/sod-skripsi/DUTS-TE.txt",
        "DUT-OMRON": "/kaggle/input/sod-skripsi/DUT-OMRON.txt"
    }

    model_paths = sorted(glob.glob(os.path.join(args.model_dir, "*.pth")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GAPNet(arch='iformer_tiny', pretrained=False).to(device)
    model.eval()

    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Dataset', 'MAE', 'F_max', 'F_weighted', 'E_mean', 'E_max', 'S_measure'])

    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)

    for epoch_path in model_paths:
        epoch_name = os.path.basename(epoch_path)
        print(f"\n==================== Memproses Model: {epoch_name} ====================")
        
        state_dict = torch.load(epoch_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            # Fix tambahan jika ada nesting backbone ganda
            if name.startswith('backbone.backbone.'):
                name = name.replace('backbone.backbone.', 'backbone.')
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=True)

        for ds_name, ds_txt in datasets.items():
            if not os.path.exists(ds_txt): continue
            
            img_gt_pairs = []
            with open(ds_txt, 'r') as f:
                for line in f:
                    img_gt_pairs.append(line.strip().split())

            # CARA INISIALISASI TERAMAN (Constructor-based)
            FM = M.FmeasureV2(handlers=[M.FmeasureHandler(with_dynamic=True, with_adaptive=True)])
            EM = M.EmeasureV2(handlers=[M.EmeasureHandler(with_dynamic=True, with_adaptive=True)])
            WFM = M.WeightedFmeasure()
            SM = M.Smeasure()
            MAE = M.MAE()

            with torch.no_grad():
                for img_rel, gt_rel in tqdm(img_gt_pairs, desc=f"{ds_name}"):
                    base_data_root = "/kaggle/input/sod-skripsi/"
                    img_path = os.path.join(base_data_root, img_rel.lstrip('/'))
                    gt_path = os.path.join(base_data_root, gt_rel.lstrip('/'))
                    
                    image = cv2.imread(img_path)
                    gt = cv2.imread(gt_path, 0)
                    if image is None or gt is None: continue
                    
                    orig_h, orig_w = image.shape[:2]
                    img_input = cv2.resize(image, (args.width, args.height))
                    img_input = img_input.astype(np.float32) / 255.
                    img_input = (img_input - mean) / std
                    img_input = img_input[:, :, ::-1].copy().transpose((2, 0, 1)) 
                    img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

                    res = model(img_tensor)
                    pred_map = res[:, 0, :, :].unsqueeze(1)
                    pred_map = F.interpolate(pred_map, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    pred_np = (pred_map.squeeze().cpu().numpy() * 255).astype(np.uint8)

                    # Pastikan binary mask GT bersih
                    if np.max(gt) > 1:
                        gt = (gt > 128).astype(np.uint8) * 255

                    FM.step(pred=pred_np, gt=gt)
                    WFM.step(pred=pred_np, gt=gt)
                    SM.step(pred=pred_np, gt=gt)
                    EM.step(pred=pred_np, gt=gt)
                    MAE.step(pred=pred_np, gt=gt)

            # --- EKSTRAKSI & PENAMAAN VARIABEL YANG KONSISTEN ---
            fm_results_final = FM.get_results()['fm'] 
            em_results_final = EM.get_results()['em']
            
            row = [
                epoch_name, 
                ds_name,
                f"{MAE.get_results()['mae']:.4f}",
                f"{fm_results_final['curve'].max():.4f}", # F-max
                f"{WFM.get_results()['wfm']:.4f}",       # F-weighted
                f"{em_results_final['curve'].mean():.4f}", # E-mean
                f"{em_results_final['curve'].max():.4f}",  # E-max
                f"{SM.get_results()['sm']:.4f}"          # S-measure
            ]

            with open(args.out_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

    print(f"\nEvaluasi selesai! Hasil disimpan di: {args.out_csv}")

if __name__ == "__main__":
    main()
