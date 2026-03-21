import os
import shutil
import random
import cv2
import torch
from models import model as net
import numpy as np
import transforms as myTransforms
from dataset import Dataset
import time
from argparse import ArgumentParser
from saleval import SalEval
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import py_sod_metrics as M


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
        assert len(torch.unique(targets)) == 3, torch.unique(targets)
        valid_mask = (targets != 255)
        
        targets_valid = targets.clone()
        targets_valid[targets == 255] = 0
        
        BCE_func = nn.BCELoss(reduction='none')
        bce = BCE_func(inputs, targets_valid)
        
        bce = bce * valid_mask
        
        bce = torch.sum(bce) / (torch.sum(valid_mask) + 1e-8)  
    else:
        assert len(torch.unique(targets)) == 2, torch.unique(targets)
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
        num_scales = 6
        assert len(inputs.shape) == 4, f"prediction has {len(inputs.shape)} dimensions: C, scale, H, W"
        assert inputs.size()[1] == num_scales, f"prediction has {inputs.size()[1]} levels of features"

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

        if self.supervision == 0:
            loss_overall = losses[:1]
        elif self.supervision == 8:
            losses[3] = criterion(inputs[:, 3, :, :], targets[:, 2, :, :], ignore_index=False)
            loss_overall = losses[:1] + losses[3:5]
        else:
            loss_overall = losses[:1] + losses[3:5]

        return sum(loss_overall)/len(loss_overall)*3


@torch.no_grad()
def val(args, val_loader, model, criterion):
    model.eval()
    sal_eval_val = SalEval()
    SM = M.Smeasure()
    EM = M.Emeasure()
    
    epoch_loss = []
    bar = tqdm(val_loader, desc="Validating")

    for iter, (input, target) in enumerate(bar):
        input_var = input.to(device)
        target_var = target.to(device).float()

        output = model(input_var)
        
        # FIX: Squeeze target_var agar dimensinya 3D [Batch, H, W]
        target_squeezed = target_var.squeeze(1) 

        # Hitung Val Loss dengan membuang dimensi channel yang tidak diperlukan
        loss = criterion.criterion(output[:, 0, :, :], target_squeezed) / args.iter_size
        epoch_loss.append(loss.item())

        # F-beta & MAE
        sal_eval_val.add_batch(output[:, 0, :, :], target_var)
        
        # S-Measure & E-Measure
        preds = (output[:, 0, :, :].cpu().numpy() * 255).astype(np.uint8)
        gts = target_squeezed.cpu().numpy().astype(np.uint8)
        
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis=0)
            gts = np.expand_dims(gts, axis=0)
            
        for i in range(preds.shape[0]):
            SM.step(preds[i], gts[i])
            EM.step(preds[i], gts[i])

        if iter % 10 == 0:
            bar.set_description(f"Val Loss: {sum(epoch_loss) / len(epoch_loss):.5f}")

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = sal_eval_val.get_metric()
    S_measure = SM.get_results()['sm']
    E_measure = EM.get_results()['em']['curve'].max()

    return average_epoch_loss_val, F_beta, MAE, S_measure, E_measure


def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    model.train()
    epoch_loss = []
    iter_time = 0
    optimizer.zero_grad()
    bar = tqdm(train_loader)
    
    for iter, (input, target) in enumerate(bar):
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)
        
        input = input.to(device)
        target = target.to(device)
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        
        if args.ms1:
            resize = np.random.choice([320, 352, 384])
            input_var = F.interpolate(input_var, size=(resize, resize), mode='bilinear', align_corners=False)
            target_var = F.interpolate(target_var.unsqueeze(dim=1), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(dim=1)
        
        output = model(input_var)
        loss = criterion(output, target_var) / args.iter_size
        loss.backward()
        
        iter_time += 1
        if iter_time % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss.append(loss.data.item())
        
        if iter % 10 == 0:
            bar.set_description("loss: {:.5f}, lr: {:.8f}".format(sum(epoch_loss) / len(epoch_loss), lr))
    
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    return average_epoch_loss_train, 0, 0, lr


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


def train_validate_saliency(args):
    model = net.GAPNet(arch=args.arch, pretrained=True)

    args.savedir = args.savedir + '/'
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    shutil.copy('scripts/train.py', args.savedir + 'train.py')
    shutil.copy('scripts/train.sh', args.savedir + 'train.sh')
    os.system("scp -r {} {}".format("scripts", args.savedir))
    os.system("scp -r {} {}".format("models", args.savedir))

    if args.gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters))

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

    val_names = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]

    trainLoader_main = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_main, process_label=True, ignore_index=args.igi),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_scale1, process_label=True, ignore_index=args.igi),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
    trainLoader_scale2 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_scale2, process_label=True, ignore_index=args.igi),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valLoader = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[0], transform=valDataset, process_label=False),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.ms:
        max_batches = len(trainLoader_main) + len(trainLoader_scale1) + len(trainLoader_scale2)
    else:
        max_batches = len(trainLoader_main)
    print('max_batches {}'.format(max_batches))
    cudnn.benchmark = True

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_file = args.savedir + args.log_file
    if os.path.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
    logger.write("\nParameters: %s" % (str(total_paramters)))
    logger.write("\n%s\t\t%s\t%s\t%s\t%s\t%s\tlr" % ('Epoch', 'Loss(Tr)', 'F_beta (tr)', 'MAE (tr)', 'F_beta (val)', 'MAE (val)'))
    logger.flush()

    if args.group_lr:
        normal_parameters = []
        picked_parameters = []
        for pname, p in model.named_parameters():
            if 'backbone' in pname:
                picked_parameters.append(p)
            else:
                normal_parameters.append(p)
        optimizer = torch.optim.Adam([
            {'params': normal_parameters, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': picked_parameters, 'lr': args.lr / 10, 'weight_decay': 1e-4},
        ], lr=args.lr, betas=(0.9, args.adam_beta2), eps=1e-08, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, args.adam_beta2), eps=1e-08, weight_decay=1e-4)
        
    cur_iter = start_epoch * max_batches
    
    if args.resume is not None and os.path.isfile(args.resume):
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded optimizer state from checkpoint")

    criteria = CrossEntropyLoss
    if args.bcedice:
        criteria = BCEDiceLoss

    criteria = CEOLoss(criterion=criteria, ignore_index=args.igi, supervision=args.supervision)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    hist_train_loss = []
    hist_val_loss = []
    hist_val_fbeta = []
    hist_val_mae = []
    hist_val_smeasure = []
    hist_val_emeasure = []

    # =========================================================================
    # SINGLE CLEAN TRAINING LOOP 
    # =========================================================================
    for epoch in range(start_epoch, args.max_epochs):
        loss_train_s1 = loss_train_s2 = 0.0
        
        # 1. Training Skala Kecil (S1 & S2)
        if args.ms:
            loss_train_s1, _, _, _ = train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale1)
            torch.cuda.empty_cache()
            
            loss_train_s2, _, _, _ = train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale2)
            torch.cuda.empty_cache()

        # 2. Training Skala Utama (Main Scale 384x384)
        loss_train_main, _, _, _ = train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
        hist_train_loss.append(loss_train_main) 
        cur_iter += len(trainLoader_main)
        torch.cuda.empty_cache()

        # 3. Evaluasi Validasi
        print(f"\nStart to evaluate on epoch {epoch+1}")
        start_time = time.time()
        loss_val, F_beta_val, MAE_val, S_m_val, E_m_val = val(args, valLoader, model, criteria)
        
        hist_val_loss.append(loss_val)
        hist_val_fbeta.append(F_beta_val)
        hist_val_mae.append(MAE_val)
        hist_val_smeasure.append(S_m_val)
        hist_val_emeasure.append(E_m_val)
        
        print(f"Elapsed evaluation time: {(time.time()-start_time)/3600.0:.4f} hours")
        torch.cuda.empty_cache()

        # 4. Simpan Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_val': loss_val,
            'iou_val': F_beta_val,
        }, args.savedir + 'checkpoint.pth.tar')

        if epoch > args.max_epochs * 0.5:
            model_file_name = args.savedir + 'model_' + str(epoch + 1) + '.pth'
            torch.save(model.state_dict(), model_file_name)

        # 5. Penulisan Log yang Bersih 
        log_str = f"Epoch {epoch+1:03d}\tTr_Loss(s1,s2,main): {loss_train_s1:.4f}, {loss_train_s2:.4f}, {loss_train_main:.4f}\tVal_Loss: {loss_val:.4f}\tF_beta: {F_beta_val:.4f}\tMAE: {MAE_val:.4f}\tS_m: {S_m_val:.4f}\tE_m: {E_m_val:.4f}"
        logger.write(log_str + "\n")
        logger.flush()
        
        print(f"Epoch {epoch+1} Results -> Train Loss: {loss_train_main:.4f} | Val Loss: {loss_val:.4f} | MAE: {MAE_val:.4f} | F_beta: {F_beta_val:.4f} | S-m: {S_m_val:.4f} | E-m: {E_m_val:.4f}\n")


    # =========================================================================
    # GENERATE PLOTS SELESAI TRAINING
    # =========================================================================
    epochs_range = range(1, len(hist_train_loss) + 1)
    
    # Plot 1: Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, hist_train_loss, 'b-', label='Train Loss (Main Scale)')
    plt.plot(epochs_range, hist_val_loss, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.savedir + 'curve_loss.png', dpi=300)
    plt.close()

    # Plot 2: MAE
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, hist_val_mae, 'g-', label='Validation MAE')
    plt.title('Validation MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.savedir + 'curve_mae.png', dpi=300)
    plt.close()

    # Plot 3: Kinerja F-beta, S-measure, E-measure
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, hist_val_fbeta, 'm-', label='F-beta')
    plt.plot(epochs_range, hist_val_smeasure, 'c-', label='S-measure')
    plt.plot(epochs_range, hist_val_emeasure, 'y-', label='E-measure')
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.savedir + 'curve_metrics.png', dpi=300)
    plt.close()
    
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data/", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./gapnet', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
    parser.add_argument('--log_file', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--iter_size', default=1, type=int)
    parser.add_argument('--arch', default='vgg16', type=str)
    parser.add_argument('--ms', default=1, type=int)  
    parser.add_argument('--ms1', default=0, type=int)
    parser.add_argument('--adam_beta2', default=0.999, type=float)  
    parser.add_argument('--bcedice', default=0, type=int)
    parser.add_argument('--group_lr', default=0, type=int)
    parser.add_argument('--gpu_id', default='0, 1', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--igi', default=0, type=int, help="ignore index")
    parser.add_argument('--supervision', default=8, type=int, help="supervision signals")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.savedir += f'seed{args.seed}-'

    print('Called with args:')
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.batch_size *= torch.cuda.device_count()
    args.num_workers *= torch.cuda.device_count()
    train_validate_saliency(args)
