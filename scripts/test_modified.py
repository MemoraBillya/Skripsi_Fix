import os
import os.path as osp
import cv2
import torch
import numpy as np
import csv
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from models import model as net
from tqdm import tqdm
import py_sod_metrics as M

@torch.no_grad()
def test(args, model, image_list, label_list, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    # Inisialisasi Metrik Global (Untuk Keseluruhan Dataset)
    SM = M.Smeasure()
    EM = M.Emeasure()
    WFM = M.WeightedFmeasure()
    FM = M.Fmeasure()
    MAE = M.MAE()

    # List untuk menyimpan skor per gambar
    per_image_scores = []

    for idx in tqdm(range(len(image_list)), desc="Testing"):
        image_name = osp.basename(image_list[idx])
        
        # Baca gambar dan label
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)

        # Proses Ground Truth
        gt = label.copy()
        if np.max(gt) == 255:
            gt = gt / 255.0

        # Pre-processing Gambar
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std
        img = img[:, :, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        # Prediksi Model
        imgs_out = model(img)
        img_out = imgs_out[:, 0, :, :].unsqueeze(dim=0)
        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        
        # Konversi ke Map Biner 0-255 untuk Metrik dan Simpan
        pred_map = (img_out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_map = (gt * 255).astype(np.uint8)

        # 1. Simpan Gambar Hasil (Saliency Map)
        cv2.imwrite(osp.join(save_dir, image_name[:-4] + '.png'), pred_map)

        # 2. Evaluasi Metrik Global
        SM.step(pred=pred_map, gt=gt_map)
        EM.step(pred=pred_map, gt=gt_map)
        WFM.step(pred=pred_map, gt=gt_map)
        FM.step(pred=pred_map, gt=gt_map)
        MAE.step(pred=pred_map, gt=gt_map)

        # 3. Evaluasi Metrik Per Gambar (S-measure & MAE)
        # Kita menggunakan S-measure dan MAE karena ini yang paling akurat 
        # untuk menilai kualitas 1 gambar tunggal secara langsung.
        img_SM = M.Smeasure()
        img_MAE = M.MAE()
        img_SM.step(pred=pred_map, gt=gt_map)
        img_MAE.step(pred=pred_map, gt=gt_map)
        
        per_image_scores.append({
            'filename': image_name,
            'S_measure': round(img_SM.get_results()['sm'], 4),
            'MAE': round(img_MAE.get_results()['mae'], 4)
        })

    # --- SIMPAN SKOR PER GAMBAR KE CSV ---
    csv_path = osp.join(save_dir, 'per_image_scores.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['filename', 'S_measure', 'MAE']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(per_image_scores)

    # --- EKSTRAK HASIL METRIK KESELURUHAN (GLOBAL) ---
    sm = SM.get_results()['sm']
    em_max = EM.get_results()['em']['curve'].max()
    em_mean = EM.get_results()['em']['curve'].mean()
    wfm = WFM.get_results()['wfm']
    fm_max = FM.get_results()['fm']['curve'].max()
    fm_mean = FM.get_results()['fm']['curve'].mean()
    mae = MAE.get_results()['mae']

    results = {
        'S_measure': sm,
        'max_E_measure': em_max,
        'mean_E_measure': em_mean,
        'max_F_measure': fm_max,
        'mean_F_measure': fm_mean,
        'w_F_measure': wfm,
        'MAE': mae
    }
    
    return results

def main(args, file_list):
    image_list = list()
    label_list = list()
    
    # Membaca daftar file (Bisa format .txt atau .lst)
    list_path = osp.join(args.data_dir, file_list + '.txt')
    if not osp.exists(list_path):
        list_path = osp.join(args.data_dir, file_list + '.lst')
        
    with open(list_path) as fid:
        for line in fid:
            line_arr = line.split()
            image_list.append(osp.join(args.data_dir, line_arr[0].strip()))
            label_list.append(osp.join(args.data_dir, line_arr[1].strip()))

    model = net.GAPNet(arch=args.arch)
    
    if not osp.isfile(args.pretrained):
        print(f"❌ Pre-trained model '{args.pretrained}' tidak ditemukan.")
        exit(-1)

    print(f"Menggunakan Model: {args.pretrained}")
    state_dict = torch.load(args.pretrained, map_location='cpu')

    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    save_dir = osp.join(args.savedir, file_list)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    results = test(args, model, image_list, label_list, save_dir)
    return results

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='vgg16', help='the backbone name')
    parser.add_argument('--data_dir', default="./data-sod", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--savedir', default='./test_outputs', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pretrained', default=None, help='Path ke file model.pth.tar')
    parser.add_argument('--igi', default=0, type=int, help="ignore index")
    parser.add_argument('--supervision', default=1, type=int)
    args = parser.parse_args()

    try:
        import py_sod_metrics
    except ImportError:
        print("WAJIB INSTALL: pip install pysodmetrics")
        exit(-1)

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # 5 Dataset Evaluasi Utama
    data_lists = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]
    
    print("\n" + "="*80)
    print("🚀 MEMULAI EVALUASI MODEL PADA BENCHMARK DATASETS")
    print("="*80)

    # Cetak Header Tabel
    print(f"{'Dataset':<12} | {'S-meas':<7} | {'max_E':<7} | {'mean_E':<7} | {'max_F':<7} | {'mean_F':<7} | {'w_F':<7} | {'MAE':<7}")
    print("-" * 80)

    all_results = {}
    
    for dataset_name in data_lists:
        res = main(args, dataset_name)
        all_results[dataset_name] = res
        
        print(f"{dataset_name:<12} | {res['S_measure']:.4f} | {res['max_E_measure']:.4f} | {res['mean_E_measure']:.4f} | {res['max_F_measure']:.4f} | {res['mean_F_measure']:.4f} | {res['w_F_measure']:.4f} | {res['MAE']:.4f}")

    print("="*80)
    print(f"✅ Saliency Maps dan CSV Skor Per-Gambar telah disimpan di: {args.savedir}")
