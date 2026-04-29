import shutil
import torch
import cv2
import time
import os
import csv
import os.path as osp
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import defaultdict
import py_sod_metrics as M
from models import model_video as net
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

@torch.no_grad()
def test_and_evaluate(args, model, video_dict, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    
    # Inisialisasi Metrik Global
    FM_global = M.Fmeasure()
    SM_global = M.Smeasure()
    MAE_global = M.MAE()

    csv_data = []

    print("\n🚀 Memulai Proses Testing & Evaluasi...")
    for video_key, frames in tqdm(video_dict.items(), desc="Memproses Video"):
        dataset_name, video_name = video_key
        
        # Inisialisasi Metrik Per-Video
        FM_vid = M.Fmeasure()
        SM_vid = M.Smeasure()
        MAE_vid = M.MAE()
        
        for img_path, flow_path, label_path in frames:
            image = cv2.imread(img_path)
            flow = cv2.imread(flow_path)
            gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            orig_shape = image.shape[:2]
            
            # Pre-processing
            img = cv2.resize(image, (args.width, args.height))
            flow_img = cv2.resize(flow, (args.width, args.height))
            
            img = img.astype(np.float32) / 255.
            img -= mean
            img /= std
            img = img[:, :, ::-1].transpose((2, 0, 1))
            img_var = torch.from_numpy(img).unsqueeze(0).to(device)
            
            flow_img = flow_img.astype(np.float32) / 255.
            flow_img -= mean
            flow_img /= std
            flow_img = flow_img[:, :, ::-1].transpose((2, 0, 1))
            flow_var = torch.from_numpy(flow_img).unsqueeze(0).to(device)
            
            # Prediksi
            preds = model(img_var, flow_var)
            pred_map = preds[:, 0, :, :].unsqueeze(1)
            
            # Post-processing & Simpan Mask
            pred_map = F.interpolate(pred_map, size=orig_shape, mode='bilinear', align_corners=False)
            pred_np = (pred_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            frame_name = osp.basename(img_path).replace('.jpg', '.png')
            vid_save_dir = osp.join(save_dir, dataset_name, video_name)
            os.makedirs(vid_save_dir, exist_ok=True)
            
            save_path = osp.join(vid_save_dir, frame_name)
            cv2.imwrite(save_path, pred_np)
            
            # Hitung Metrik
            FM_vid.step(pred=pred_np, gt=gt)
            SM_vid.step(pred=pred_np, gt=gt)
            MAE_vid.step(pred=pred_np, gt=gt)
            
            FM_global.step(pred=pred_np, gt=gt)
            SM_global.step(pred=pred_np, gt=gt)
            MAE_global.step(pred=pred_np, gt=gt)
            
        # Catat Metrik Video
        fm_v = FM_vid.get_results()['fm']['curve'].max()
        sm_v = SM_vid.get_results()['sm']
        mae_v = MAE_vid.get_results()['mae']
        csv_data.append([dataset_name, video_name, sm_v, fm_v, mae_v])

    # Catat Metrik Global
    fm_g = FM_global.get_results()['fm']['curve'].max()
    sm_g = SM_global.get_results()['sm']
    mae_g = MAE_global.get_results()['mae']
    csv_data.append(['ALL_DATASETS', 'GLOBAL_AVERAGE', sm_g, fm_g, mae_g])

    # Simpan CSV
    csv_path = osp.join(args.save_dir, f"{args.method_tag}_metrics.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Video', 'S_m', 'F_max', 'MAE'])
        writer.writerows(csv_data)
        
    print(f"\n✅ Rekap metrik CSV berhasil disimpan ke: {csv_path}")
    return sm_g, fm_g, mae_g

def main(args):
    video_dict = defaultdict(list)
    total_frames = 0
    
    lst_file = osp.join(args.data_dir, args.dataset_name + '.lst')
    with open(lst_file) as fid:
        for line in fid:
            line_arr = line.strip().split()
            if len(line_arr) >= 3:
                img_path = osp.join(args.data_dir, line_arr[0].strip())
                flow_path = osp.join(args.data_dir, line_arr[1].strip())
                gt_path = osp.join(args.data_dir, line_arr[2].strip())
                
                rel_path = osp.relpath(img_path, args.data_dir)
                parts = rel_path.replace('\\', '/').split('/')
                
                try:
                    imgs_idx = parts.index('Imgs')
                    dataset_name = parts[0]
                    video_name = parts[imgs_idx - 1]
                except ValueError:
                    dataset_name = parts[0]
                    video_name = parts[-2]
                
                video_dict[(dataset_name, video_name)].append((img_path, flow_path, gt_path))
                total_frames += 1

    print(f"📊 Menemukan {total_frames} frames dari {len(video_dict)} video unik.")

    model = net.GAPNet(arch=args.arch, pretrained=False)
    
    if not osp.isfile(args.pretrained):
        print(f'❌ File pre-trained tidak ditemukan: {args.pretrained}')
        exit(-1)

    state_dict = torch.load(args.pretrained, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # FLOPs & FPS
    dummy_img = torch.rand(1, 3, args.width, args.height).to(device)
    dummy_flow = torch.rand(1, 3, args.width, args.height).to(device)
    flops = FlopCountAnalysis(model, (dummy_img, dummy_flow))
    print(f"⚙️  Total FLOPs: {flops.total()/1e9:.4f} GMac")

    print("⚡ Menguji kecepatan FPS (Batch Size 1)...")
    bs = 1
    with torch.no_grad():
        for _ in range(50): _ = model(dummy_img, dummy_flow)
        from time import time
        start_t = time()
        for _ in range(100): _ = model(dummy_img, dummy_flow)
        total_t = time() - start_t
    print(f"⏱️  Kecepatan: {100 / total_t * bs:.2f} FPS")

    # Testing & Evaluasi
    save_dir = osp.join(args.save_dir, args.method_tag)
    sm_g, fm_g, mae_g = test_and_evaluate(args, model, video_dict, save_dir)
    print(f'\n🎯 Hasil Akhir Global -> S_m: {sm_g:.4f}, F_max: {fm_g:.4f}, MAE: {mae_g:.4f}')
    
    # ZIP
    zip_path = osp.join(args.save_dir, args.method_tag)
    print(f"📦 Mengompresi hasil prediksi ke {zip_path}.zip ...")
    shutil.make_archive(zip_path, 'zip', save_dir)
    print("✅ Kompresi selesai!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='iformer_tiny')
    parser.add_argument('--data_dir', default="/kaggle/input/datasets/billydawson/vsod-skripsi-lst/")
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--dataset_name', default='DAVSOD_test')
    parser.add_argument('--save_dir', default='/kaggle/working/test_results')
    parser.add_argument('--method_tag', default='gapnet_iformer')
    args = parser.parse_args()

    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
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
