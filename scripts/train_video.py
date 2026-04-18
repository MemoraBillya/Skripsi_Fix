import os
import torch
from models import model_video as net
import numpy as np
import transforms as myTransforms
from dataset import Dataset
import time
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import py_sod_metrics as M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    if bce < 0:
        print(bce)
        exit()
    else:
        return bce

class CEOLoss(nn.Module):
    def __init__(self, criterion=BCEDiceLoss, ignore_index=False, supervision=0):
        super(CEOLoss, self).__init__()
        self.criterion = criterion
        self.ignore_index = ignore_index
        self.supervision = supervision

    def forward(self, inputs, targets):
        criterion = self.criterion
        losses = []
        for i in [0, 1, 2, 5]:
            dt = inputs[:, i, :, :]
            gt = targets[:, i, :, :]
            losses.append(criterion(dt, gt, ignore_index=False))
        for i in range(3, 5):
            dt = inputs[:, i, :, :]
            gt = targets[:, i, :, :]
            losses.insert(i, criterion(dt, gt, ignore_index=self.ignore_index))

        losses[3] = criterion(inputs[:, 3, :, :], targets[:, 2, :, :], ignore_index=False)
        loss_overall = losses[:1] + losses[3:5]
        return sum(loss_overall)/len(loss_overall)*3

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

@torch.no_grad()
def val_monitoring(val_loader, model, criterion):
    model.eval()
    SM = M.Smeasure()
    FM = M.Fmeasure()
    MAE_metric = M.MAE()
    epoch_loss = []
    
    for iter, (input, input_1, target) in enumerate(val_loader):
        input = input.to(device)
        input_1 = input_1.to(device)
        target = target.to(device).float()

        # Output dari model memiliki 6 channel (multi-scale)
        output = model(input, input_1)
        
        # Target disesuaikan ukurannya menjadi (Batch, H, W)
        target_squeezed = target.squeeze(1) 
        
        # --- PERHITUNGAN VALIDATION LOSS MASK TUNGGAL ---
        # Kita hanya mengambil channel ke-0 (output paling utama/resolusi penuh)
        # dan menghitung Binary Cross Entropy (BCE) terhadap mask asli.
        val_loss_single = F.binary_cross_entropy(output[:, 0, :, :], target_squeezed)
        epoch_loss.append(val_loss_single.item())
        # ------------------------------------------------
        
        preds = (output[:, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        gts = (target_squeezed.cpu().numpy() * 255).astype(np.uint8)
        
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis=0)
            gts = np.expand_dims(gts, axis=0)
            
        for i in range(preds.shape[0]):
            SM.step(preds[i], gts[i])
            FM.step(preds[i], gts[i])
            MAE_metric.step(preds[i], gts[i])

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    s_m = SM.get_results()['sm']
    f_max = FM.get_results()['fm']['curve'].max()
    mae = MAE_metric.get_results()['mae']
    return avg_loss, s_m, f_max, mae

    # def val_monitoring(val_loader, model, criterion):
#     model.eval()
#     SM = M.Smeasure()
#     FM = M.Fmeasure()
#     MAE_metric = M.MAE()
#     epoch_loss = []
    
#     for iter, (input, input_1, target) in enumerate(val_loader):
#         input = input.to(device)
#         input_1 = input_1.to(device)
#         target = target.to(device).float()

#         output = model(input, input_1)
#         loss = criterion(output, target)
#         epoch_loss.append(loss.item())

#         target_squeezed = target.squeeze(1) 
        
#         preds = (output[:, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
#         gts = (target_squeezed.cpu().numpy() * 255).astype(np.uint8)
        
#         if len(preds.shape) == 2:
#             preds = np.expand_dims(preds, axis=0)
#             gts = np.expand_dims(gts, axis=0)
            
#         for i in range(preds.shape[0]):
#             SM.step(preds[i], gts[i])
#             FM.step(preds[i], gts[i])
#             MAE_metric.step(preds[i], gts[i])

#     avg_loss = sum(epoch_loss) / len(epoch_loss)
#     s_m = SM.get_results()['sm']
#     f_max = FM.get_results()['fm']['curve'].max()
#     mae = MAE_metric.get_results()['mae']
#     return avg_loss, s_m, f_max, mae

def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    model.train()
    
    if args.freeze_bn:
        model.apply(freeze_bn)
        
    epoch_loss = []
    iter_time = 0
    optimizer.zero_grad()
    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for iter, (input, input_1, target) in enumerate(bar):
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)
        
        input = input.to(device)
        input_1 = input_1.to(device)
        target = target.to(device).float()
        
        if args.ms1:
            resize = np.random.choice([320, 352, 384])
            input = F.interpolate(input, size=(resize, resize), mode='bilinear', align_corners=False)
            input_1 = F.interpolate(input_1, size=(resize, resize), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(resize, resize), mode='bilinear', align_corners=False)
            # target = F.interpolate(target.unsqueeze(dim=1), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(dim=1)
        
        output = model(input, input_1)
        loss = criterion(output, target) / args.iter_size
        loss.backward()
        
        iter_time += 1
        if iter_time % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss.append(loss.data.item())
        
        if iter % 10 == 0:            
            bar.set_description("loss: {:.5f}, lr: {:.8f}".format(sum(epoch_loss) / len(epoch_loss), lr))
    
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    return average_epoch_loss_train, lr

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200: 
        lr = args.lr * 0.99 * (iter + 1) / 200 + 0.01 * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_video_saliency(args):
    model = net.GAPNet(arch=args.arch, pretrained=True)
    os.makedirs(args.savedir, exist_ok=True)
    
    total_parameters = sum(p.numel() for p in model.parameters())
    print("\n" + "="*50)
    print(f"🧠 Total Model Parameters: {total_parameters:,}")
    print("="*50 + "\n")

    if args.gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)), 
                        np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))] 

    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(7./224.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(320, 320),
        myTransforms.RandomCropResize(int(7./224.*320)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(352, 352),
        myTransforms.RandomCropResize(int(7./224.*352)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.ToTensor()
    ])

    trainLoader_main = torch.utils.data.DataLoader(
        Dataset(args.data_dir, args.dataset_list, transform=trainDataset_main, process_label=True, ignore_index=args.igi, use_flow=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, args.dataset_list, transform=trainDataset_scale1, process_label=True, ignore_index=args.igi, use_flow=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, args.dataset_list, transform=trainDataset_scale2, process_label=True, ignore_index=args.igi, use_flow=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    has_val = False
    if args.val_dataset and args.val_dataset.strip() != "":
        has_val = True
        valLoader = torch.utils.data.DataLoader(
            Dataset(args.data_dir, args.val_dataset, transform=valDataset, process_label=False, use_flow=True),
            batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.ms:
        max_batches = len(trainLoader_main) + len(trainLoader_scale1) + len(trainLoader_scale2)
    else:
        max_batches = len(trainLoader_main)
    cudnn.benchmark = True
    start_epoch = 0
    cur_iter = 0

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, args.adam_beta2), eps=1e-08, weight_decay=1e-4)

    # 1. RESUME LOGIC (Bulletproof)
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"=> Melanjutkan Training Video dari checkpoint: '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cur_iter = start_epoch * max_batches # Sinkronisasi Poly LR
        print(f"=> Berhasil memuat checkpoint (Melanjutkan dari Epoch {start_epoch + 1})")
        
    # 2. FINE-TUNING IMAGE MODEL LOGIC
    elif args.pretrained_model is not None and os.path.isfile(args.pretrained_model):
        print(f"🔄 Melakukan Fine-Tuning dari Model Image SOD: {args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model, map_location='cpu')
        actual_state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        
        cleaned_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in actual_state_dict.items()}
        
        model_dict = model.state_dict()
        for k, v in cleaned_state_dict.items():
            if k in model_dict:
                model_dict[k] = v
            # Copy bobot RGB ke Optical Flow
            k_flow = k.replace('backbone.', 'backbone_1.')
            if k_flow in model_dict:
                model_dict[k_flow] = v
                
        model.load_state_dict(model_dict, strict=False)

    log_file = os.path.join(args.savedir, args.log_file)
    logger = open(log_file, 'a' if os.path.isfile(log_file) else 'w')
    criteria = CEOLoss(criterion=CrossEntropyLoss if not args.bcedice else BCEDiceLoss, ignore_index=args.igi, supervision=args.supervision)

    hist_tr_loss = []
    hist_s = []
    hist_f = []
    hist_mae = []
    hist_val_loss = []

    print("\n🚀 MEMULAI TRAINING VIDEO")
    if args.freeze_bn:
        print("❄️  BatchNorm Statistics: FROZEN")
    print(f"📦 Gradient Accumulation: {args.iter_size} (Effective Batch: {args.batch_size * args.iter_size})")
    print("="*50)

    # for epoch in range(start_epoch, args.max_epochs):
    #     # 1. Training
    #     loss_train, current_lr = train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
    #     hist_tr_loss.append(loss_train)
    #     cur_iter += len(trainLoader_main)
    #     torch.cuda.empty_cache()

    #     log_str = f"Epoch {epoch+1:02d} | Tr_Loss: {loss_train:.4f} | LR: {current_lr:.8f}"

    for epoch in range(start_epoch, args.max_epochs):
        # 1. Training
        if args.ms:
            # Training pada skala 1 (320x320)
            _, current_lr = train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale1)
            torch.cuda.empty_cache()
            
            # Training pada skala 2 (352x352)
            _, current_lr = train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale2)
            torch.cuda.empty_cache()

        # Training pada skala utama (384x384)
        loss_train, current_lr = train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
        hist_tr_loss.append(loss_train)
        cur_iter += len(trainLoader_main)
        torch.cuda.empty_cache()

        log_str = f"Epoch {epoch+1:02d} | Tr_Loss: {loss_train:.4f} | LR: {current_lr:.8f}"

        # 2. Validation (Jika ada dataset)
        if has_val:
            print(f"Menguji Validasi Epoch {epoch+1}...")
            v_loss, s, f, m = val_monitoring(valLoader, model, criteria)
            hist_val_loss.append(v_loss)
            hist_s.append(s); hist_f.append(f); hist_mae.append(m)
            log_str += f" || Val -> Loss:{v_loss:.4f} S:{s:.3f} Fmax:{f:.3f} MAE:{m:.3f}"
            torch.cuda.empty_cache()

        # 3. Logging & Checkpoint
        print("✅ " + log_str)
        logger.write(log_str + "\n")
        logger.flush()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.savedir, 'checkpoint.pth.tar'))

        if epoch >= 5:
            torch.save(model.state_dict(), os.path.join(args.savedir, f"model_{epoch+1}.pth"))

    logger.close()

    # ==========================================================
    # PEMBUATAN GRAFIK 
    # ==========================================================
    epochs_range_tr = range(start_epoch + 1, start_epoch + len(hist_tr_loss) + 1)
    
    # Plot Training Loss
    # Plot Training vs Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range_tr, hist_tr_loss, 'k-', linewidth=2, label='Training Loss')
    if has_val and len(hist_val_loss) > 0:
        plt.plot(epochs_range_val, hist_val_loss, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.savedir, 'curve_loss.png'), dpi=300)
    plt.close()

    # Plot Metrik Validasi (Jika ada)
    if has_val and len(hist_s) > 0:
        epochs_range_val = range(start_epoch + 1, start_epoch + len(hist_s) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range_val, hist_s, 'b-', label='S-measure')
        plt.plot(epochs_range_val, hist_f, 'r--', label='Max F-measure')
        plt.title('Validation Metrics over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.savedir, 'curve_val_metrics.png'), dpi=300)
        plt.close()
        
    print("🎉 SELURUH PROSES TRAINING SELESAI!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data/", help='Data directory')
    parser.add_argument('--dataset_list', default="DAVSOD_DAVIS_train", help='Comma separated list for training')
    parser.add_argument('--val_dataset', default="", help='Validation dataset list (empty to skip validation)')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=30, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size') 
    parser.add_argument('--step_loss', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.7e-6, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly')
    parser.add_argument('--savedir', default='/kaggle/working/gapnet_video/', help='Directory to save')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--log_file', default='trainLog.txt')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--iter_size', default=1, type=int) 
    parser.add_argument('--freeze_bn', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--arch', default='iformer_tiny', type=str)
    parser.add_argument('--pretrained_model', default=None, type=str) 
    parser.add_argument('--ms', default=0, type=int)  
    parser.add_argument('--ms1', default=1, type=int) 
    parser.add_argument('--adam_beta2', default=0.99, type=float)
    parser.add_argument('--bcedice', default=1, type=int)
    parser.add_argument('--group_lr', default=0, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--igi', default=0, type=int)
    parser.add_argument('--supervision', default=8, type=int)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if args.gpu else 'cpu')
    args.batch_size *= torch.cuda.device_count()
    args.num_workers *= torch.cuda.device_count()
    train_video_saliency(args)


# import os
# import shutil
# import random
# import cv2
# import torch
# from models import model_video as net
# import numpy as np
# import transforms as myTransforms
# from dataset import Dataset
# # from parallel import DataParallelModel, DataParallelCriterion
# import time
# from argparse import ArgumentParser
# from saleval import SalEval
# import torch.backends.cudnn as cudnn
# import torch.optim.lr_scheduler
# # from torch.nn.parallel import gather
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm


# def BCEDiceLoss(inputs, targets, ignore_index=False):
#     bce = CrossEntropyLoss(inputs, targets, ignore_index)
#     inter = (inputs * targets).sum()
#     eps = 1e-5
#     dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
#     # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
#     return bce + 1 - dice


# def CrossEntropyLoss(inputs, targets, ignore_index=False):
#     index = [i for i in range(targets.size()[0]) if not torch.all(targets[i] == 10)]

#     targets = targets[index, :, :]
#     inputs = inputs[index, :, :]

#     if ignore_index:
#         assert len(torch.unique(targets)) == 3, torch.unique(targets)
#         valid_mask = (targets != 255)
        
#         targets_valid = targets.clone()
#         targets_valid[targets == 255] = 0
        
#         BCE_func = nn.BCELoss(reduction='none')
#         bce = BCE_func(inputs, targets_valid)
        
#         bce = bce * valid_mask
        
#         bce = torch.sum(bce) / (torch.sum(valid_mask) + 1e-8)  # Add small epsilon to avoid division by zero
#     else:
#         assert len(torch.unique(targets)) == 2, torch.unique(targets)
#         bce = F.binary_cross_entropy(inputs, targets)

#     if bce < 0:
#         print(bce)
#         exit()
#     else:
#         return bce


# class CEOLoss(nn.Module):
#     def __init__(self, criterion=BCEDiceLoss, ignore_index=False, supervision=0):
#         super(CEOLoss, self).__init__()
#         self.criterion = criterion
#         self.ignore_index = ignore_index
#         self.supervision = supervision

#     def forward(self, inputs, targets):
#         # num_scales = 6 if self.dds else 1
#         num_scales = 6
#         assert len(inputs.shape) == 4, f"prediction has {len(inputs.shape)} (should be 4) dimensions: C, scale, H, W"
#         assert inputs.size()[1] == num_scales, f"prediction has {inputs.size()[1]} (should be {num_scales}) levels of features"
#         # assert target.size()[1] == num, f"target has {target.size()[1]} (should be 3) levels of features"

#         criterion = self.criterion

#         losses = []
#         for i in [0, 1, 2, 5]:
#             dt = inputs[:, i, :, :]
#             gt = targets[:, i, :, :]
#             losses.append(criterion(dt, gt, ignore_index=False))

#         for i in range(3, 5):
#             dt = inputs[:, i, :, :]
#             gt = targets[:, i, :, :]
#             losses.insert(i, criterion(dt, gt, ignore_index=self.ignore_index))

#         losses[3] = criterion(inputs[:, 3, :, :], targets[:, 2, :, :], ignore_index=False)
#         loss_overall = losses[:1] + losses[3:5]

#         return sum(loss_overall)/len(loss_overall)*3


# @torch.no_grad()
# def val(args, val_loader, model, criterion):
#     # switch to evaluation mode
#     model.eval()
#     sal_eval_val = SalEval()
#     epoch_loss = []
#     total_batches = len(val_loader)

#     for iter, (input, input_1, target) in enumerate(tqdm(val_loader)):
#         start_time = time.time()
#         input = input.to(device)
#         input_1 = input_1.to(device)
#         target = target.to(device)

#         input_var = torch.autograd.Variable(input)
#         input_1_var = torch.autograd.Variable(input_1)
#         target_var = torch.autograd.Variable(target).float()

#         # run the model
#         output = model(input_var, input_1_var)

#         time_taken = time.time() - start_time

#         epoch_loss.append(0)
#         sal_eval_val.add_batch(output[:, 0, :, :],  target_var)

#     average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
#     F_beta, MAE = sal_eval_val.get_metric()

#     return average_epoch_loss_val, F_beta, MAE


# def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
#     # switch to train mode
#     model.train()
#     # sal_eval_train = SalEval()
#     epoch_loss = []
#     total_batches = len(train_loader)
#     iter_time = 0
#     optimizer.zero_grad()
#     bar = tqdm(train_loader)
    
#     num_iterations = 0
    
#     for iter, (input, input_1, target) in enumerate(bar):
#         iter_start_time = time.time()
#         data_loading_time = iter_start_time
#         num_iterations += 1
                
#         lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)
        
#         input = input.to(device)
#         input_1 = input_1.to(device)
#         target = target.to(device)
        
#         input_var = torch.autograd.Variable(input)
#         input_1_var = torch.autograd.Variable(input_1)
#         target_var = torch.autograd.Variable(target).float()
        
#         if args.ms1:
#             resize = np.random.choice([320, 352, 384])
#             input_var = F.interpolate(input_var, size=(resize, resize), mode='bilinear', align_corners=False)
#             input_1_var = F.interpolate(input_1_var, size=(resize, resize), mode='bilinear', align_corners=False)
#             target_var = F.interpolate(target_var.unsqueeze(dim=1), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(dim=1)
        
#         output = model(input_var, input_1_var)
#         loss = criterion(output, target_var) / args.iter_size
#         loss.backward()
        
#         iter_time += 1
#         if iter_time % args.iter_size == 0:
#             optimizer.step()
#             optimizer.zero_grad()
        
#         epoch_loss.append(loss.data.item())
        
#         # Update progress bar with timing info
#         if iter % 10 == 0:            
#             bar.set_description("loss: {:.5f}, lr: {:.8f}".format(sum(epoch_loss) / len(epoch_loss), lr))
    
    
#     average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
#     F_beta, MAE = 0, 0  # sal_eval_train.get_metric()

#     return average_epoch_loss_train, F_beta, MAE, lr


# def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
#     if args.lr_mode == 'step':
#         lr = args.lr * (0.1 ** (epoch // args.step_loss))
#     elif args.lr_mode == 'poly':
#         max_iter = max_batches * args.max_epochs
#         lr = args.lr * (1 - iter * 1.0 / max_iter) ** 0.9
#     else:
#         raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
#     if epoch == 0 and iter < 200:  # warm up
#         lr = args.lr * 0.99 * (iter + 1) / 200 + 0.01 * args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


# def train_validate_saliency(args):
#     model = net.GAPNet(arch=args.arch, pretrained=True)

#     args.savedir = args.savedir + '/'
#     # create the directory if not exist
#     if not os.path.exists(args.savedir):
#         os.makedirs(args.savedir)

#     print('copying train scripts to snapshots dir')
#     shutil.copy('scripts/train_video.py', args.savedir + 'train_video.py')
#     shutil.copy('scripts/train_video.sh', args.savedir + 'train_video.sh')
#     os.system("scp -r {} {}".format("scripts", args.savedir))
#     os.system("scp -r {} {}".format("models", args.savedir))

#     if args.gpu and torch.cuda.device_count() > 1:
#         # model = nn.DataParallel(model)
#         model = nn.DataParallel(model)

#     model = model.to(device)

#     total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
#     print('Total network parameters: ' + str(total_paramters))

#     NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)),  # MEAN, BGR
#                         np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))]  # STD, BGR

#     # compose the data with transforms
#     trainDataset_main = myTransforms.Compose([
#         myTransforms.Normalize(*NORMALISE_PARAMS),
#         myTransforms.Scale(args.width, args.height),
#         myTransforms.RandomCropResize(int(7./224.*args.width)),
#         myTransforms.RandomFlip(),
#         # myTransforms.GaussianNoise(),
#         myTransforms.ToTensor()
#     ])

#     trainDataset_scale1 = myTransforms.Compose([
#         myTransforms.Normalize(*NORMALISE_PARAMS),
#         myTransforms.Scale(320, 320),
#         myTransforms.RandomCropResize(int(7./224.*320)),
#         myTransforms.RandomFlip(),
#         myTransforms.ToTensor()
#     ])
#     trainDataset_scale2 = myTransforms.Compose([
#         myTransforms.Normalize(*NORMALISE_PARAMS),
#         myTransforms.Scale(352, 352),
#         myTransforms.RandomCropResize(int(7./224.*352)),
#         myTransforms.RandomFlip(),
#         myTransforms.ToTensor()
#     ])

#     valDataset = myTransforms.Compose([
#         myTransforms.Normalize(*NORMALISE_PARAMS),
#         myTransforms.Scale(args.width, args.height),
#         myTransforms.ToTensor()
#     ])

#     val_names = ["DAVIS_test", "DAVSOD_test", "FBMS_test", "ViSal_test", "SegTrack-V2_test"]

#     trainLoader_main = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, 'DAVSOD_DAVIS_train', transform=trainDataset_main, process_label=True, ignore_index=args.igi,use_flow=True),
#         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,  pin_memory=True, drop_last=True)

#     trainLoader_scale1 = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, 'DAVSOD_DAVIS_train', transform=trainDataset_scale1, process_label=True, ignore_index=args.igi, use_flow=True),
#         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
#     trainLoader_scale2 = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, 'DAVSOD_DAVIS_train', transform=trainDataset_scale2, process_label=True, ignore_index=args.igi, use_flow=True),
#         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

#     valLoader = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, val_names[0], transform=valDataset, process_label=False, use_flow=True),
#         batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#     valLoader1 = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, val_names[1], transform=valDataset, process_label=False, use_flow=True),
#         batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#     valLoader2 = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, val_names[2], transform=valDataset, process_label=False, use_flow=True),
#         batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#     valLoader3 = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, val_names[3], transform=valDataset, process_label=False, use_flow=True),
#         batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#     valLoader4 = torch.utils.data.DataLoader(
#         Dataset(args.data_dir, val_names[4], transform=valDataset, process_label=False, use_flow=True),
#         batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)

#     if args.ms:
#         max_batches = len(trainLoader_main) + len(trainLoader_scale1) + len(trainLoader_scale2)
#     else:
#         max_batches = len(trainLoader_main)
#     print('max_batches {}'.format(max_batches))
#     cudnn.benchmark = True

#     start_epoch = 0

#     if args.pretrained_model is not None:
#         print("loading SOD pretrained:", args.pretrained_model)
#         temp = torch.load(args.pretrained_model, map_location='cpu')
#         model.load_state_dict(temp, strict=False)

#     if args.resume is not None:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             checkpoint = torch.load(args.resume)
#             start_epoch = checkpoint['epoch']
#             # args.lr = checkpoint['lr']
#             model.load_state_dict(checkpoint['state_dict'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))

#     log_file = args.savedir + args.log_file
#     if os.path.isfile(log_file):
#         logger = open(log_file, 'a')
#     else:
#         logger = open(log_file, 'w')
#     logger.write("\nParameters: %s" % (str(total_paramters)))
#     logger.write("\n%s\t\t%s\t%s\t%s\t%s\t%s\tlr" % ('Epoch',
#                                                      'Loss(Tr)', 'F_beta (tr)', 'MAE (tr)', 'F_beta (val)', 'MAE (val)'))
#     logger.flush()

#     normal_parameters = []
#     picked_parameters = []
#     if args.group_lr:
#         # use smaller lr in backbone
#         for pname, p in model.named_parameters():
#             if 'backbone' in pname and 'backbone_1' not in pname:
#                 picked_parameters.append(p)
#                 #print("lr/10", pname)
#             else:
#                 normal_parameters.append(p)
#         optimizer = torch.optim.Adam([
#             {
#                 'params': normal_parameters,
#                 'lr': args.lr,
#                 'weight_decay': 1e-4
#             },
#             {
#                 'params': picked_parameters,
#                 'lr': args.lr / 10,
#                 'weight_decay': 1e-4
#             },
#         ],
#             lr=args.lr,
#             betas=(0.9, args.adam_beta2),
#             eps=1e-08,
#             weight_decay=1e-4)
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, args.adam_beta2), eps=1e-08, weight_decay=1e-4)
#     cur_iter = 0

#     criteria = CrossEntropyLoss
#     if args.bcedice:
#         criteria = BCEDiceLoss
#         print("use dice loss")

#     criteria = CEOLoss(criterion=criteria, ignore_index=args.igi, supervision=args.supervision)

#     if args.gpu and torch.cuda.device_count() > 1:
#         print("using mutliple gpus")

#     epoch_idxes = []
#     F_beta_vals = []
#     F_beta_val1s = []
#     F_beta_val2s = []
#     F_beta_val3s = []
#     F_beta_val4s = []
#     F_beta_val5s = []
#     MAE_vals = []
#     MAE_val1s = []
#     MAE_val2s = []
#     MAE_val3s = []
#     MAE_val4s = []
#     MAE_val5s = []

#     for epoch in range(start_epoch, args.max_epochs):
#         # train for one epoch
#         if args.ms:
#             train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
#             cur_iter += len(trainLoader_scale1)
#             torch.cuda.empty_cache()
#             train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
#             cur_iter += len(trainLoader_scale2)
#             torch.cuda.empty_cache()

#         train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
#         cur_iter += len(trainLoader_main)
#         torch.cuda.empty_cache()

#         # evaluate on validation set
#         print("start to evaluate on epoch {}".format(epoch+1))
#         import time
#         start_time = time.time()
#         loss_val, F_beta_val, MAE_val = val(args, valLoader, model, criteria)
#         torch.cuda.empty_cache()
#         if epoch >= 0:#> args.max_epochs * 0.5:
#             loss_val1, F_beta_val1, MAE_val1 = val(args, valLoader1, model, criteria)
#             torch.cuda.empty_cache()
#             loss_val2, F_beta_val2, MAE_val2 = val(args, valLoader2, model, criteria)
#             torch.cuda.empty_cache()
#             loss_val3, F_beta_val3, MAE_val3 = val(args, valLoader3, model, criteria)
#             torch.cuda.empty_cache()
#             loss_val4, F_beta_val4, MAE_val4 = val(args, valLoader4, model, criteria)
#             torch.cuda.empty_cache()
#             # loss_val5, F_beta_val5, MAE_val5 = val(args, valLoader5, model, criteria)
#             F_beta_val5, MAE_val5 = 0, 0
#             F_beta_vals.append(F_beta_val)
#             F_beta_val1s.append(F_beta_val1)
#             F_beta_val2s.append(F_beta_val2)
#             F_beta_val3s.append(F_beta_val3)
#             F_beta_val4s.append(F_beta_val4)
#             #F_beta_val5s.append(F_beta_val5)
#             MAE_vals.append(MAE_val)
#             MAE_val1s.append(MAE_val1)
#             MAE_val2s.append(MAE_val2)
#             MAE_val3s.append(MAE_val3)
#             MAE_val4s.append(MAE_val4)
#             #MAE_val5s.append(MAE_val5)
#             epoch_idxes.append(epoch+1)

#         #     print("elapsed evaluation time: {} hours".format((time.time()-start_time)/3600.0))
#         torch.cuda.empty_cache()

#         torch.save({
#             'epoch': epoch + 1,
#             'arch': str(model),
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'loss_val': loss_val,
#             'f_val': F_beta_val,
#         }, args.savedir + 'checkpoint.pth.tar')

#         # save the model also

#         if epoch >= 5:
#             model_file_name = args.savedir + 'model_' + str(epoch + 1) + '.pth'
#             print("saving state dict to {}".format(model_file_name))
#             torch.save(model.state_dict(), model_file_name)

#         log_str = "\n{} {:.4f} {:.4f}".format(epoch+1, F_beta_val, MAE_val)
#         try:
#             log_str = log_str + " {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
#                 F_beta_val1, MAE_val1, F_beta_val2, MAE_val2, F_beta_val3, MAE_val3, F_beta_val4, MAE_val4)
#         except:
#             pass
        
#         logger.write(log_str)
#         logger.flush()
#         print("Epoch " + str(epoch) + ': Details')
#         print("Epoch No. %d: \t Val Loss = %.4f\t MAE Loss = %.4f\t F_beta(val) = %.4f\n"
#              % (epoch+1, loss_val, MAE_val, F_beta_val))
#         torch.cuda.empty_cache()
#     logger.close()


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--data_dir', default="./data/", help='Data directory')
#     parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
#     parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
#     parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
#     parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
#     parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
#     parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
#     parser.add_argument('--savedir', default='./results_vit_ea/', help='Directory to save the results')
#     parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
#     parser.add_argument('--log_file', default='trainValLog.txt', help='File that stores the training and validation logs')
#     parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
#                         help='Run on CPU or GPU. If TRUE, then GPU.')
#     parser.add_argument('--iter_size', default=1, type=int)
#     parser.add_argument('--arch', default='mobilenetv2', type=str)
#     parser.add_argument('--pretrained_model', default='pretrained/gapnet.pth', type=str)
#     parser.add_argument('--ms', default=1, type=int)  # normal multi-scale training
#     # hybrid multi-scale training. It has comparable performance with normal multi-scale training in my experiments. But I think hybrid multi-scale training may be a better choice.
#     parser.add_argument('--ms1', default=0, type=int)
#     parser.add_argument('--adam_beta2', default=0.999, type=float)  # The value of 0.99 can introduce slightly higher performance (0.1%~0.2%)
#     parser.add_argument('--bcedice', default=0, type=int)
#     parser.add_argument('--group_lr', default=0, type=int)
#     parser.add_argument('--gpu_id', default='0, 1', type=str)
#     parser.add_argument('--seed', default=42, type=int)
#     parser.add_argument('--igi', default=0, type=int, help="ignore index")
#     parser.add_argument('--supervision', default=8, type=int, help="supervision signals")
#     # parser.add_argument('--dilation_opt', default=1, choices=[1, 2], type=int, help="dilation ratios")
#     args = parser.parse_args()

#     seed = args.seed

#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True


#     print('Called with args:')
#     print(args)
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#     if args.gpu:
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#     # print(torch.cuda.device_count())
#     args.batch_size *= torch.cuda.device_count()
#     args.num_workers *= torch.cuda.device_count()
#     train_validate_saliency(args)
