import shutil
import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from models import model_video as net
from tqdm import tqdm
import pandas as pd
import py_sod_metrics as M
from fvcore.nn import FlopCountAnalysis


@torch.no_grad()
def test_and_evaluate(args, model, video_groups, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    # Inisialisasi Metrik Global (Untuk seluruh dataset)
    SM_global = M.Smeasure()
    FM_global = M.Fmeasure()
    MAE_global = M.MAE()

    results_list = []

    # Iterasi per Video
    for video_folder, paths in tqdm(video_groups.items(), desc="Testing Videos"):
        # Inisialisasi Metrik Lokal (Khusus untuk 1 video ini saja)
        SM_video = M.Smeasure()
        FM_video = M.Fmeasure()
        MAE_video = M.MAE()

        for img_path, flow_path, label_path in zip(paths['imgs'], paths['flows'], paths['labels']):
            # 1. Load data
            image = cv2.imread(img_path)
            flow = cv2.imread(flow_path)
            label = cv2.imread(label_path, 0)

            # 2. Resize
            img_resized = cv2.resize(image, (args.width, args.height))
            flow_resized = cv2.resize(flow, (args.width, args.height))
            
            # 3. Normalize Image
            img_tensor = img_resized.astype(np.float32) / 255.
            img_tensor -= mean
            img_tensor /= std
            img_tensor = img_tensor[:, :, ::-1].copy().transpose((2, 0, 1))
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
            
            # 4. Normalize Flow
            flow_tensor = flow_resized.astype(np.float32) / 255.
            flow_tensor -= mean
            flow_tensor /= std
            flow_tensor = flow_tensor[:, :, ::-1].copy().transpose((2, 0, 1))
            flow_tensor = torch.from_numpy(flow_tensor).unsqueeze(0).to(device)

            # 5. Forward Pass Model
            imgs_out = model(img_tensor, flow_tensor)
            img_out = imgs_out[:, 0, :, :].unsqueeze(dim=0)
            img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
            
            sal_map = (img_out * 255).data.cpu().numpy()[0, 0].astype(np.uint8)
            
            # 6. Simpan Masking per Dataset & per Video
            rel_path = osp.relpath(img_path, args.data_dir)
            save_rel_path = rel_path.replace('.jpg', '.png').replace('.jpeg', '.png')
            save_path = osp.join(save_dir, save_rel_path)
            
            # Buat foldernya jika belum ada
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, sal_map)
            
            # 7. Update Metrik (Video & Global)
            # Pastikan label ground truth berada dalam rentang 0-255 uint8
            if label.max() == 1:
                label = label * 255
            gt_np = label.astype(np.uint8)
            pred_np = sal_map
            
            SM_video.step(pred_np, gt_np)
            FM_video.step(pred_np, gt_np)
            MAE_video.step(pred_np, gt_np)

            SM_global.step(pred_np, gt_np)
            FM_global.step(pred_np, gt_np)
            MAE_global.step(pred_np, gt_np)

        # 8. Hitung Nilai Akhir Metrik untuk Video Saat Ini
        video_sm = SM_video.get_results()['sm']
        video_fm = FM_video.get_results()['fm']['curve'].max()
        video_mae = MAE_video.get_results()['mae']

        results_list.append({
            'Folder Video': video_folder,
            'S-measure': round(video_sm, 4),
            'Max F-measure': round(video_fm, 4),
            'MAE': round(video_mae, 4)
        })

    # 9. Hitung Nilai Akhir Metrik Global (Satu Dataset Penuh)
    global_sm = SM_global.get_results()['sm']
    global_fm = FM_global.get_results()['fm']['curve'].max()
    global_mae = MAE_global.get_results()['mae']

    results_list.append({
        'Folder Video': 'OVERALL_DATASET',
        'S-measure': round(global_sm, 4),
        'Max F-measure': round(global_fm, 4),
        'MAE': round(global_mae, 4)
    })

    return results_list, global_sm, global_fm, global_mae


def main(args):
    # 1. Baca list dataset dan kelompokkan per video
    video_groups = OrderedDict()
    
    lst_file = osp.join(args.data_dir, args.dataset_name + '.lst')
    with open(lst_file) as fid:
        for line in fid:
            line_arr = line.strip().split()
            if len(line_arr) >= 3:
                img_rel = line_arr[0].strip()
                flow_rel = line_arr[1].strip()
                gt_rel = line_arr[2].strip()
                
                img_path = osp.join(args.data_dir, img_rel)
                flow_path = osp.join(args.data_dir, flow_rel)
                gt_path = osp.join(args.data_dir, gt_rel)
                
                # Menggunakan nama folder sebagai identifier video
                video_folder = osp.dirname(img_rel) 
                
                if video_folder not in video_groups:
                    video_groups[video_folder] = {'imgs': [], 'flows': [], 'labels': []}
                
                video_groups[video_folder]['imgs'].append(img_path)
                video_groups[video_folder]['flows'].append(flow_path)
                video_groups[video_folder]['labels'].append(gt_path)

    total_frames = sum([len(v['imgs']) for v in video_groups.values()])
    print(f"Memuat {total_frames} frame dari {len(video_groups)} video.")

    # 2. Load Model
    model = net.GAPNet(arch=args.arch, pretrained=True)
    
    if not osp.isfile(args.pretrained):
        print(f'File pretrained model tidak ditemukan: {args.pretrained}')
        exit(-1)

    state_dict = torch.load(args.pretrained, map_location='cpu')
    # Jika checkpoint dari pelatihan DataParallel mengandung 'module.', kita bersihkan
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    total_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total parameter jaringan: {total_parameters/1e6:.6f}M')

    # 3. Hitung FLOPs
    print("Menghitung FLOPs...")
    flops = FlopCountAnalysis(model, (torch.rand(1, 3, args.width, args.height).to(device), 
                                     torch.rand(1, 3, args.width, args.height).to(device)))
    print(f"Total FLOPs: {flops.total()/1e9:.4f} G")

    # 4. Speed Test (FPS)
    print("Menjalankan speed test (FPS)...")
    bs = 10  # Batch size untuk test FPS
    x1 = torch.randn(bs, 3, args.width, args.height).to(device)
    x2 = torch.randn(bs, 3, args.width, args.height).to(device)
    
    with torch.no_grad():
        # Pemanasan GPU
        for _ in range(50):
            _ = model(x1, x2)
    
        total_t = 0
        iterations = 100
        for _ in range(iterations):
            start = time.time()
            _ = model(x1, x2)
            torch.cuda.synchronize() # Sinkronisasi akurat untuk GPU
            total_t += time.time() - start

    fps = (iterations * bs) / total_t
    print(f"Kecepatan (FPS): {fps:.2f} frames per second")
    print("-" * 50)
    
    # 5. Persiapkan Direktori Penyimpanan
    save_dir = osp.join(args.save_dir, args.method_tag)
    os.makedirs(save_dir, exist_ok=True)

    # 6. Jalankan Proses Testing & Evaluasi
    results_list, global_sm, global_fm, global_mae = test_and_evaluate(args, model, video_groups, save_dir)
    
    # 7. Simpan Hasil ke CSV
    csv_save_path = osp.join(save_dir, f'Metrics_{args.dataset_name}.csv')
    df = pd.DataFrame(results_list)
    df.to_csv(csv_save_path, index=False)
    
    print("-" * 50)
    print(f"✅ Evaluasi Selesai untuk Dataset: {args.dataset_name}")
    print(f"✅ Masking berhasil disimpan di: {save_dir}")
    print(f"✅ Laporan CSV berhasil disimpan di: {csv_save_path}")
    print(f"📊 HASIL OVERALL -> S-Measure: {global_sm:.4f} | Max F-measure: {global_fm:.4f} | MAE: {global_mae:.4f}")

    # ==========================================
    # INI UNTUK ZIP
    # ==========================================
    print("\n📦 Sedang mengompres hasil menjadi ZIP...")
    zip_filename = osp.join(args.save_dir, f"{args.method_tag}_{args.dataset_name}")
    shutil.make_archive(zip_filename, 'zip', save_dir)
    print(f"✅ Berhasil! File zip tersimpan di: {zip_filename}.zip")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='mobilenetv2', help='backbone: vgg16, resnet50, mobilenetv2, dll')
    parser.add_argument('--data_dir', default="./data", help='Direktori utama data')
    parser.add_argument('--width', type=int, default=384, help='Lebar gambar RGB')
    parser.add_argument('--height', type=int, default=384, help='Tinggi gambar RGB')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'), help='Gunakan GPU')
    parser.add_argument('--pretrained', default='model_10.pth', help='Path ke file model hasil training (.pth)')
    parser.add_argument('--dataset_name', default='DAVSOD_test', type=str, help="Nama dataset (nama file .lst)")
    parser.add_argument('--save_dir', default='./test_results', type=str, help="Direktori utama penyimpanan prediksi")
    parser.add_argument('--method_tag', default='GAPNet_Ours', type=str, help="Nama sub-folder eksperimen (misal: eksperimen_1)")
    
    # Argumen arsitektur bawaan GAPNet
    parser.add_argument('--dds', default=1, type=int)
    parser.add_argument('--gbg', default=1, type=int)
    parser.add_argument('--igi', default=1, type=int)
    parser.add_argument('--kvc', default=0, type=int)
    parser.add_argument('--qc', default=1, type=int)
    parser.add_argument('--attention', default="EA", type=str)
    parser.add_argument('--dilation_opt', default=1, type=int)
    parser.add_argument('--low_global_vit', default=0, type=int)
    parser.add_argument('--vit_dwconv', default=1, type=int)
    parser.add_argument('--supervision', default=8, type=int)
    
    args = parser.parse_args()

    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    main(args)

# import shutil
# import torch
# import cv2
# import time
# import os
# import os.path as osp
# import numpy as np
# from torch.autograd import Variable
# import torch.nn.functional as F
# from argparse import ArgumentParser
# from collections import OrderedDict
# from saleval import SalEval
# from models import model_video as net
# from tqdm import tqdm
# import random
# from fvcore.nn import FlopCountAnalysis


# @torch.no_grad()
# def test(args, model, image_list, flow_list, label_list, save_dir):
#     mean = [0.406, 0.456, 0.485]
#     std = [0.225, 0.224, 0.229]
#     eval = SalEval()

#     for idx in tqdm(range(len(image_list))):
#         # Load image, flow, and label
#         image = cv2.imread(image_list[idx])
#         flow = cv2.imread(flow_list[idx])
#         label = cv2.imread(label_list[idx], 0)
#         label = label / 255

#         # Resize the image and flow to specified dimensions
#         img = cv2.resize(image, (args.width, args.height))
#         flow_img = cv2.resize(flow, (args.width, args.height))
        
#         # Normalize image
#         img = img.astype(np.float32) / 255.
#         img -= mean
#         img /= std
        
#         # Normalize flow (same normalization as image)
#         flow_img = flow_img.astype(np.float32) / 255.
#         flow_img -= mean
#         flow_img /= std

#         # Convert BGR to RGB and transpose
#         img = img[:, :, ::-1].copy()
#         img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(img).unsqueeze(0)
#         img = Variable(img)
        
#         flow_img = flow_img[:, :, ::-1].copy()
#         flow_img = flow_img.transpose((2, 0, 1))
#         flow_img = torch.from_numpy(flow_img).unsqueeze(0)
#         flow_img = Variable(flow_img)
        
#         label = torch.from_numpy(label).float().unsqueeze(0)

#         img = img.to(device)
#         flow_img = flow_img.to(device)
#         label = label.to(device)

#         # Run the two-stream model with image and flow inputs
#         imgs_out = model(img, flow_img)
#         img_out = imgs_out[:, 0, :, :].unsqueeze(dim=0)
#         img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
#         sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        
#         # Create save path following the specified pattern
#         # Original: DAVSOD/test/select_0606/Imgs/0398.jpg
#         # Save as: {save_dir}/{method_tag}/DAVSOD/select_0606/0398.png
#         rel_path = osp.relpath(image_list[idx], args.data_dir)
#         # Remove 'test/' and 'Imgs/' from the path and change extension to .png
#         save_rel_path = rel_path.replace('/test/', '/').replace('/Imgs/', '/').replace('.jpg', '.png')
#         save_path = osp.join(save_dir, save_rel_path)
        
#         # Create directory if it doesn't exist
#         save_path_dir = osp.dirname(save_path)
#         if not osp.exists(save_path_dir):
#             os.makedirs(save_path_dir, exist_ok=True)
            
#         cv2.imwrite(save_path, sal_map)
        
#         # Add to evaluation
#         eval.add_batch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

#     F_beta_max, MAE = eval.get_metric()
#     return F_beta_max, MAE


# def main(args):
#     # Read the dataset list file
#     image_list = list()
#     flow_list = list()
#     label_list = list()
    
#     lst_file = osp.join(args.data_dir, args.dataset_name + '.lst')
#     with open(lst_file) as fid:
#         for line in fid:
#             line_arr = line.strip().split()
#             if len(line_arr) >= 3:
#                 image_list.append(osp.join(args.data_dir, line_arr[0].strip()))
#                 flow_list.append(osp.join(args.data_dir, line_arr[1].strip()))
#                 label_list.append(osp.join(args.data_dir, line_arr[2].strip()))

#     print(f"Loaded {len(image_list)} test samples")

#     # Load the two-stream model
#     model = net.GAPNet(arch=args.arch, pretrained=True)
    
#     # Load pretrained weights
#     if not osp.isfile(args.pretrained):
#         print(f'Pre-trained model file does not exist: {args.pretrained}')
#         exit(-1)

#     state_dict = torch.load(args.pretrained, map_location='cpu')
    
#     total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
#     print(f'Total network parameters: {total_paramters/1e6:.6f}M')

#     model.load_state_dict(state_dict, strict=True)
#     model = model.to(device)
#     model.eval()

#     # Calculate FLOPs
#     flops = FlopCountAnalysis(model, (torch.rand(1, 3, 384, 384).to(device), 
#                                      torch.rand(1, 3, 384, 384).to(device)))
#     print(f"total flops: {flops.total()/1e9:.4f}G")

#     # Speed test
#     print("Running speed test...")
#     bs = 20
#     x1 = torch.randn(bs, 3, 384, 384).to(device)
#     x2 = torch.randn(bs, 3, 384, 384).to(device)
    
#     # Warm up
#     with torch.no_grad():
#         for _ in range(50):
#             y = model(x1, x2)
    
#         from time import time
#         total_t = 0
#         for _ in range(100):
#             start = time()
#             y = model(x1, x2)
#             total_t += time() - start

#     print("FPS", 100 / total_t * bs)
#     print(f"PyTorch batchsize={bs} speed test completed")
    
#     # Create save directory
#     save_dir = osp.join(args.save_dir, args.method_tag)
#     if not osp.isdir(save_dir):
#         os.makedirs(save_dir, exist_ok=True)

#     # Run testing
#     F_beta_max, MAE = test(args, model, image_list, flow_list, label_list, save_dir)
#     print(f'Dataset: {args.dataset_name}')
#     print(f'F_beta_max: {F_beta_max:.4f}, MAE: {MAE:.4f}')
    
#     return F_beta_max, MAE


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--arch', default='mobilenetv2', help='the backbone name of EDN, vgg16, resnet50, or mobilenetv2')
#     parser.add_argument('--data_dir', default="./data", help='Data directory')
#     parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
#     parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
#     parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
#                         help='Run on CPU or GPU. If TRUE, then GPU')
#     parser.add_argument('--pretrained', default=None, help='Path to pretrained model file (.pth)')
#     parser.add_argument('--dds', default=1, type=int, help="diverse supervision")
#     parser.add_argument('--gbg', default=1, type=int, help="global guidance")
#     parser.add_argument('--igi', default=1, type=int, help="ignore index")
#     parser.add_argument('--kvc', default=0, type=int, help="concatenate x1, x2 for k, v computation or not")
#     parser.add_argument('--qc', default=1, type=int, help="concatenate x1, x2 for q computation or not")
#     parser.add_argument('--attention', default="EA", choices=["EA", "SA"], type=str, help="attention mechanisms: self-attention, efficient-attention")
#     parser.add_argument('--dilation_opt', default=1, choices=[1, 2], type=int, help="dilation option")
#     parser.add_argument('--low_global_vit', default=0, type=int, help="use vit for edge/global feature fusion")
#     parser.add_argument('--vit_dwconv', default=1, type=int, help="add dwconv in vit ffn")
#     parser.add_argument('--supervision', default=8, type=int, help="supervision signals")
#     parser.add_argument('--dataset_name', default='DAVSOD_test', type=str, help="dataset name for .lst file")
#     parser.add_argument('--save_dir', default='./test_results', type=str, help="directory to save predictions")
#     parser.add_argument('--method_tag', default='video_mobilenetv2', type=str, help="method tag for save path")
    
#     args = parser.parse_args()

#     print('Called with args:')
#     print(args)

#     device = torch.device('cuda') if args.gpu else torch.device('cpu')
    
#     F_beta_max, MAE = main(args)
#     print(f'Final Results - F_beta_max: {F_beta_max:.4f}, MAE: {MAE:.4f}')
