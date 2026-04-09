import os
import os.path as osp
import cv2
import torch
import numpy as np
import csv
import time
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from models import model as net
from tqdm import tqdm
import py_sod_metrics as M

# WAJIB IMPORT INI UNTUK MENGHITUNG FLOPS (Bawaan asli kode Anda)
from fvcore.nn import FlopCountAnalysis

@torch.no_grad()
def test(args, model, image_list, label_list, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    # Inisialisasi Metrik Global
    SM = M.Smeasure()
    EM = M.Emeasure()
    WFM = M.WeightedFmeasure()
    FM = M.Fmeasure()
    MAE = M.MAE()

    per_image_scores = []

    for idx in tqdm(range(len(image_list)), desc="Testing"):
        image_name = osp.basename(image_list[idx])
        
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)

        gt = label.copy()
        if np.max(gt) == 255:
            gt = gt / 255.0

        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std
        img = img[:, :, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        imgs_out = model(img)
        img_out = imgs_out[:, 0, :, :].unsqueeze(dim=0)
        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        
        pred_map = (img_out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_map = (gt * 255).astype(np.uint8)

        # 1. Simpan Gambar Mask (Saliency Map)
        cv2.imwrite(osp.join(save_dir, image_name[:-4] + '.png'), pred_map)

        # 2. Update Metrik Global
        SM.step(pred=pred_map, gt=gt_map)
        EM.step(pred=pred_map, gt=gt_map)
        WFM.step(pred=pred_map, gt=gt_map)
        FM.step(pred=pred_map, gt=gt_map)
        MAE.step(pred=pred_map, gt=gt_map)

        # 3. Hitung Metrik Per Gambar (Khusus S-measure & MAE)
        img_SM = M.Smeasure()
        img_MAE = M.MAE()
        img_SM.step(pred=pred_map, gt=gt_map)
        img_MAE.step(pred=pred_map, gt=gt_map)
        
        per_image_scores.append({
            'filename': image_name,
            'S_measure': round(img_SM.get_results()['sm'], 4),
            'MAE': round(img_MAE.get_results()['mae'], 4)
        })

    # Simpan Skor Per Gambar ke CSV
    csv_path = osp.join(save_dir, 'per_image_scores.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['filename', 'S_measure', 'MAE']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_scores)

    # Ekstrak Skor Global Akhir
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

def main(args, file_list, is_first_run=False):
    image_list = list()
    label_list = list()
    
    list_path = osp.join(args.data_dir, file_list + '.txt')
    if not osp.exists(list_path):
        list_path = osp.join(args.data_dir, file_list + '.lst')
        
    with open(list_path) as fid:
        # for line in fid:
        #     line_arr = line.split()
        #     image_list.append(osp.join(args.data_dir, line_arr[0].strip()))
        #     label_list.append(osp.join(args.data_dir, line_arr[1].strip()))
        for line in fid:
            line_arr = line.split()
            # FIX: Hapus '/' berlebih di awal string agar path tersambung sempurna
            img_path = args.data_dir.rstrip('/') + '/' + line_arr[0].strip().lstrip('/')
            lbl_path = args.data_dir.rstrip('/') + '/' + line_arr[1].strip().lstrip('/')
            image_list.append(img_path)
            label_list.append(lbl_path)

    model = net.GAPNet(arch=args.arch)
    
    if not osp.isfile(args.pretrained):
        print(f"❌ Pre-trained model '{args.pretrained}' tidak ditemukan.")
        exit(-1)

    if is_first_run:
        print(f"Menggunakan Model: {args.pretrained}")
        
    state_dict = torch.load(args.pretrained, map_location='cpu')

    # if 'state_dict' in state_dict:
    #     model.load_state_dict(state_dict['state_dict'], strict=True)
    # else:
    #     model.load_state_dict(state_dict, strict=True)

    # 1. Ambil state_dict asli (baik dari checkpoint maupun file model langsung)
    if 'state_dict' in state_dict:
        actual_state_dict = state_dict['state_dict']
    else:
        actual_state_dict = state_dict

    # 2. Bersihkan prefix 'module.' bawaan DataParallel (Multi-GPU Training)
    cleaned_state_dict = {}
    for k, v in actual_state_dict.items():
        # Jika nama kunci berawalan 'module.', potong 7 karakter pertamanya
        name = k[7:] if k.startswith('module.') else k 
        cleaned_state_dict[name] = v

    # 3. Load bobot yang sudah dibersihkan ke dalam model
    model.load_state_dict(cleaned_state_dict, strict=True)

    model = model.to(device)
    model.eval()

    # =========================================================================
    # PERHITUNGAN FLOPS & FPS (HANYA DIJALANKAN SEKALI DI AWAL)
    # =========================================================================
    if is_first_run:
        print("\n" + "="*50)
        print("⚙️  MENGHITUNG KOMPLEKSITAS MODEL (FLOPs & FPS)")
        print("="*50)
        
        # 1. Hitung FLOPs
        flops = FlopCountAnalysis(model, torch.rand(1, 3, args.width, args.height).to(device))
        print(f"Total FLOPs : {flops.total()/1e9:.4f} G")

        # 2. Hitung FPS
        bs = 20
        x = torch.randn(bs, 3, args.width, args.height).to(device)
        print("Melakukan Pemanasan (Warm-up) GPU...")
        for _ in range(50):
            _ = model(x)
            
        print("Menghitung Kecepatan Inference...")
        total_t = 0
        for _ in range(100):
            start = time.time()
            _ = model(x)
            if args.gpu:
                torch.cuda.synchronize() # Sinkronisasi GPU agar hitungan waktu presisi
            total_t += time.time() - start

        fps = 100 / total_t * bs
        print(f"Batch Size  : {bs}")
        print(f"Kecepatan   : {fps:.2f} FPS")
        print("="*50 + "\n")

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
        
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("WAJIB INSTALL: pip install fvcore")
        exit(-1)

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    data_lists = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]
    
    print("\n" + "="*85)
    print("🚀 MEMULAI EVALUASI MODEL PADA BENCHMARK DATASETS")
    print("="*85)

    print(f"{'Dataset':<12} | {'S-meas':<7} | {'max_E':<7} | {'mean_E':<7} | {'max_F':<7} | {'mean_F':<7} | {'w_F':<7} | {'MAE':<7}")
    print("-" * 85)

    all_results = {}
    
    for idx, dataset_name in enumerate(data_lists):
        # Flag 'True' hanya untuk iterasi pertama agar FPS dihitung sekali
        is_first = (idx == 0) 
        res = main(args, dataset_name, is_first_run=is_first)
        all_results[dataset_name] = res
        
        print(f"{dataset_name:<12} | {res['S_measure']:.4f} | {res['max_E_measure']:.4f} | {res['mean_E_measure']:.4f} | {res['max_F_measure']:.4f} | {res['mean_F_measure']:.4f} | {res['w_F_measure']:.4f} | {res['MAE']:.4f}")

    print("="*85)
    print(f"✅ Saliency Maps dan CSV Skor Per-Gambar telah disimpan di: {args.savedir}")

# -------------------------------------
# ORIGINAL SCRIPT
# -------------------------------------

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
# from models import model as net
# from tqdm import tqdm
# # from train2 import gt2gt_ms
# import random
# from fvcore.nn import FlopCountAnalysis


# @torch.no_grad()
# def test(args, model, image_list, label_list, save_dir):

#     mean = [0.406, 0.456, 0.485]
#     std = [0.225, 0.224, 0.229]
#     eval = SalEval()

#     for idx in tqdm(range(len(image_list))):
#         # for idx in tqdm(range(1)):
#         image = cv2.imread(image_list[idx])
#         label = cv2.imread(label_list[idx], 0)
#         label = label / 255

#         # resize the image to 1024x512x3 as in previous papers
#         img = cv2.resize(image, (args.width, args.height))
#         img = img.astype(np.float32) / 255.
#         img -= mean
#         img /= std

#         img = img[:, :, ::-1].copy()
#         img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(img).unsqueeze(0)
#         img = Variable(img)
#         # print(img.size())
#         label = torch.from_numpy(label).float().unsqueeze(0)

#         img = img.to(device)
#         label = label.to(device)

#         num_areas = 6 if args.dds else 1
#         # full map, edge, center, center+others, edge+others, others
#         #areas = [#['_full_map', '_edge', '_center', '_center_other', '_edge_other', '_high_global']
#         imgs_out = model(img)
#         img_out = imgs_out[:, 0, :, :].unsqueeze(dim=0)
#         img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
#         sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
#         cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] +'.png'), sal_map)
#         # for i, area in enumerate(areas[0:num_areas]):
#         #     img_out = imgs_out[:, i, :, :].unsqueeze(dim=0)
#         #     # print(idx, i, torch.max(img_out), torch.min(img_out), torch.mean(img_out))
#         #     img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
#         #     sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
#         #     # gt_map = (label.unsqueeze(dim=0)*255).data.cpu().numpy()[0, 0].astype(np.uint8)
#         #     # print(np.min(gt_map), np.max(gt_map), gt_map.shape())
#         #     # print(osp.basename(image_list[idx])[-8:-4])
#         #     if osp.basename(image_list[idx])[-12:-4] in ['00000003', '00000023', '00000025']:
#         #         cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + area + '.png'), sal_map)
#         #         # cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '_gtgray.png'), gt_map)
#         #         # shutil.copy(image_list[idx], osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '_gtcolor.png'))
#         #    if i == 0:
#         eval.add_batch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

#     F_beta_max, MAE = eval.get_metric()
#     # print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))
#     return F_beta_max, MAE


# def main(args, file_list):
#     # read all the images in the folder
#     image_list = list()
#     label_list = list()
#     with open(args.data_dir + '/' + file_list + '.txt') as fid:
#         for line in fid:
#             line_arr = line.split()
#             image_list.append(args.data_dir + '/' + line_arr[0].strip())
#             label_list.append(args.data_dir + '/' + line_arr[1].strip())

#     # model = net.GAPNet(arch=args.arch, global_guidance=args.gbg, diverse_supervision=args.dds, attention=args.attention, kv_conc=args.kvc)
#     model = net.GAPNet(arch=args.arch)
#     if not osp.isfile(args.pretrained):
#         print('Pre-trained model file does not exist...')
#         exit(-1)

#     state_dict = torch.load(args.pretrained, map_location='cpu')

#     total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
#     # total_parameters_saved = sum(p.numel() for p in state_dict.values())
#     print(f'Total network parameters: {total_paramters/1e6:.6f}M')
#     # print('Total saved network parameters: ' + str(total_parameters_saved))

#     model.load_state_dict(state_dict, strict=True)

#     model = model.to(device)
#     model.eval()
#     # set to evaluation mode

#     flops = FlopCountAnalysis(model, torch.rand(1, 3, 384, 384).to(device))
#     print(f"total flops: {flops.total()/1e9:.4f}G")

#     ######################################
#     #### PyTorch Test [BatchSize 20] #####
#     ######################################
#     bs = 20
#     x = torch.randn(bs, 3, 384, 384).to(device)
#     for _ in range(50):
#         # warm up
#         y = model(x)
#     from time import time
#     total_t = 0
#     for _ in range(100):
#         start = time()
#         y = model(x)
#         # p = p + 1 # replace torch.cuda.synchronize()
#         total_t += time() - start

#     print("FPS", 100 / total_t * bs)
#     print(f"PyTorch batchsize={bs} speed test completed, expected 450FPS for RTX 3090!")
#     save_dir = osp.join(folder, file_list)
#     if not osp.isdir(save_dir):
#         os.makedirs(save_dir)

#     F_beta_max, MAE = test(args, model, image_list, label_list, save_dir)
#     return F_beta_max, MAE


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--arch', default='vgg16', help='the backbone name of EDN, vgg16, resnet50, or mobilenetv2')
#     parser.add_argument('--data_dir', default="./data-sod", help='Data directory')
#     parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
#     parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
#     parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
#     parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
#                         help='Run on CPU or GPU. If TRUE, then GPU')
#     parser.add_argument('--pretrained', default=None, help='Pretrained model')
#     parser.add_argument('--add_dwconv', default=0, type=int)
#     parser.add_argument('--last_channel', default=80, type=int)
#     #parser.add_argument('--low_scale', default=8, type=int)
#     parser.add_argument('--dds', default=1, type=int, help="diverse supervision")
#     parser.add_argument('--gbg', default=1, type=int, help="global guidance")
#     parser.add_argument('--igi', default=0, type=int, help="ignore index")
#     parser.add_argument('--kvc', default=0, type=int, help="concatenate x1, x2 for k, v computation or not")
#     parser.add_argument('--qc', default=1, type=int, help="concatenate x1, x2 for q computation or not")
#     parser.add_argument('--attention', default="EA", choices=["EA", "SA"], type=str, help="attention mechanisms: self-attention, efficient-attention")
#     parser.add_argument('--dilation_opt', default=1, choices=[1, 2], type=int, help="dilation option")
#     parser.add_argument('--low_global_vit', default=0, type=int, help="use vit for edge/global feature fusion")
#     parser.add_argument('--vit_dwconv', default=1, type=int, help="add dwconv in vit ffn")
#     parser.add_argument('--supervision', default=1, choices=[0, 1, 2, 3, 4, 5], type=int, help="supervision signals")
#     args = parser.parse_args()

#     print('Called with args:')
#     print(args)

#     # data_lists1 = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]
#     data_lists1 = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S", "SOC6K", "THUR15K"]
#     # data_lists2 = ["THUR15K"]
#     # data_lists = ["DUTS-TE"]
#     folder = args.savedir
#     device = torch.device('cuda') if args.gpu else torch.device('cpu')

#     F_max_list, F_mean_list, MAE_list = [], [], []
    
#     for data_index, data_list in enumerate(data_lists1):
#         print("processing ", data_list)
#         epoch_best = 30
#         print(f"best epoch for {data_list} is: {epoch_best}")
#         F_max, MAE = main(args, data_list)
#         F_max_list.append(F_max)
#         MAE_list.append(MAE)
#         #F_mean_list.append(F_mean)
#         print(F_max_list, MAE_list)
#         #args.pretrained = folder
