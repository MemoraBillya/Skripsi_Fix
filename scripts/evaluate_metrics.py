%%writefile /kaggle/working/Skripsi_Fix/scripts/evaluate_metrics.py
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
import py_sod_metrics as M
from models import model as net

@torch.no_grad()
def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Memuat model: {args.pretrained}")
    model = net.GAPNet(arch='iformer_tiny', pretrained=False)
    
    state_dict = torch.load(args.pretrained, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    model.eval()

    image_list = []
    label_list = []
    with open(args.val_list, 'r') as f:
        for line in f:
            img_path, gt_path = line.strip().split()
            image_list.append(os.path.join(args.data_dir, img_path))
            label_list.append(os.path.join(args.data_dir, gt_path))
            
    print(f"Total citra validasi: {len(image_list)}")

    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)

    print("Memproses evaluasi citra, mohon tunggu...")
    for idx in range(len(image_list)):
        image = cv2.imread(image_list[idx])
        gt = cv2.imread(label_list[idx], cv2.IMREAD_GRAYSCALE)
        
        orig_shape = image.shape[:2]
        
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img = (img - mean) / std
        img = img[:, :, ::-1].copy()
        img = img.transpose((2, 0, 1))
        
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        
        preds = model(img_tensor)
        pred_map = preds[:, 0, :, :].unsqueeze(1)
        
        pred_map = F.interpolate(pred_map, size=orig_shape, mode='bilinear', align_corners=False)
        pred_np = (pred_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        FM.step(pred=pred_np, gt=gt)
        WFM.step(pred=pred_np, gt=gt)
        SM.step(pred=pred_np, gt=gt)
        EM.step(pred=pred_np, gt=gt)
        MAE.step(pred=pred_np, gt=gt)

    fm_results = FM.get_results()['fm']
    em_results = EM.get_results()['em']
    
    results = {
        'F_max': fm_results['curve'].max(),
        'F_w': WFM.get_results()['wfm'],
        'MAE': MAE.get_results()['mae'],
        'S_m': SM.get_results()['sm'],
        'E_max': em_results['curve'].max(),
        'E_mean': em_results['curve'].mean()
    }
    
    print("\n" + "="*50)
    print(f"HASIL EVALUASI: {os.path.basename(args.pretrained)}")
    print("="*50)
    print(f"F-measure (max) [F beta max] : {results['F_max']:.4f}")
    print(f"Weighted F-measure [F w]     : {results['F_w']:.4f}")
    print(f"MAE                          : {results['MAE']:.4f}")
    print(f"S-measure                    : {results['S_m']:.4f}")
    print(f"Max E-measure                : {results['E_max']:.4f}")
    print(f"Mean E-measure               : {results['E_mean']:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="/kaggle/working/data/", help='Data directory')
    parser.add_argument('--val_list', default="/kaggle/working/data/DUTS-TE.lst", help='Path to validation txt/lst')
    parser.add_argument('--pretrained', required=True, help='Path to .pth file')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    args = parser.parse_args()
    
    evaluate_model(args)
