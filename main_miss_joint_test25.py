#GT沒有點座標就不計算、只對振幅做去噪
#使用DwT、embedding用線性層、把graph改成coco(已經考慮對稱的結構)、GCN鄰接矩陣用yolo
#針對不同部分給予不同權重(test3是用RMSE倒數，test5是用RMSE)、DwT改成沿著時間做去噪
#調整權重beta值(beta=1)、把training set 裡面看起來與人體偏差很多的資料拿掉
#loss function改成smooth L1、權重也用成smooth L1
#51維轉17維:51把維轉成1維，再複製成17維丟進mlpgcn
#用Env1訓練
import os
import glob
import torch
import random
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from common.utils import *
from common.camera import *
import common.eval_cal as eval_cal
from common.arguments import parse_args

from common.load_data_hm36 import Fusion
from common.load_data_3dhp import Fusion_3dhp
from common.h36m_dataset import Human36mDataset
from common.mpi_inf_3dhp_dataset import Mpi_inf_3dhp_Dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from model.block.refine import post_refine, refine_model
from model.graphmlp_origin import Model
#from model.graphmlp_test1 import Model
from scipy.signal import butter, filtfilt
import pywt

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def weighted_smooth_l1_loss(predicted, target, joint_weights, beta=1.0):
    """
    predicted, target: (B, J, 2)
    joint_weights: (J,)
    beta: Smooth L1 的分界點
    """
    mask = (target.abs().sum(dim=-1) > 0).float()     # (B,J)
    diff = (predicted - target).abs()                 # (B,J,2)

    smooth = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    ).mean(dim=-1)  # (B,J) -> 平均 (x,y)

    weighted = smooth * joint_weights[None, :] * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom


def per_joint_smooth_l1(predicted, target, beta=1.0):
    """
    計算每個關節的平均 Smooth L1 誤差，忽略 target=(0,0)。
    predicted, target: (N,J,2)
    return: (J,) tensor
    """
    mask = (target.abs().sum(dim=-1) > 0)  # (N,J)
    diff = (predicted - target).abs()      # (N,J,2)

    smooth = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    ).mean(dim=-1)  # (N,J)

    joint_smooth = torch.zeros(predicted.shape[1], device=predicted.device)
    for j in range(predicted.shape[1]):
        valid = mask[:, j]
        if valid.sum() > 0:
            joint_smooth[j] = smooth[valid, j].mean()
        else:
            joint_smooth[j] = 0.0
    return joint_smooth



def make_joint_weights(smooth_err, beta=1.0, eps=1e-6):
    """
    使用 Smooth L1 誤差作為每個關節的權重依據。
    smooth_err: (J,) tensor
    """
    normed = (smooth_err + eps) / (smooth_err.mean() + eps)
    weights = normed ** beta
    weights = weights / weights.mean()
    weights = weights.clamp(min=0.5, max=2.0)
    return weights



def avg_pck_result(predicted, target, alpha=0.2, left_shoulder=6, right_hip=11):
    """
    PCK@alpha，忽略 target 為 (0,0) 的關節；若肩點缺失，該樣本不計。
    """
    assert predicted.shape == target.shape
    B, J, _ = predicted.shape

    # 樣本是否肩關節皆有效
    ls_valid = (target[:, left_shoulder].abs().sum(dim=-1) > 0)
    rs_valid = (target[:, right_hip].abs().sum(dim=-1) > 0)
    sample_valid = ls_valid & rs_valid                     # [B]

    # 肩寬（避免除 0）
    shoulder_dists = torch.norm(
        target[:, left_shoulder] - target[:, right_hip], dim=-1
    ).clamp(min=1e-6)                                      # [B]

    dists = torch.norm(predicted - target, dim=-1)         # [B,J]
    normalized = dists / shoulder_dists[:, None]           # [B,J]

    # 關節有效（非 (0,0)）且樣本肩寬有效
    joint_valid = (target.abs().sum(dim=-1) > 0) & sample_valid[:, None]  # [B,J]
    if joint_valid.sum() == 0:
        return 0.0

    correct = (normalized <= alpha) & joint_valid
    return (correct.float().sum() / joint_valid.float().sum()).item()

def train(dataloader, model, model_refine, optimizer, epoch, joint_weights=None):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader, 0)):
        if args.dataset == 'csi':
            input_2D, gt_2D = data
            input_2D = input_2D.cuda()
            gt_2D = gt_2D.cuda()
            output_2D = model(input_2D)

            if joint_weights is None:
                # 沒有權重 → 普通 Smooth L1
                criterion = nn.SmoothL1Loss(beta=1.0)
                loss = criterion(output_2D, gt_2D)
            else:
                # 有權重 → 使用自訂 Smooth L1 權重版
                loss = weighted_smooth_l1_loss(output_2D, gt_2D, joint_weights, beta=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
        torch.cuda.empty_cache()

    return loss_all['loss'].avg


def test(actions, dataloader, model, model_refine):
    model.eval()

    if args.dataset == 'csi':
        total_pck = 0
        count = 0
        losses = []
        all_pred, all_gt = [], []

        for data in tqdm(dataloader, 0):
            input_2D, gt_2D = data
            input_2D = input_2D.cuda()
            gt_2D = gt_2D.cuda()
            output_2D = model(input_2D)

            loss = eval_cal.mpjpe(output_2D, gt_2D)
            losses.append(loss.item())

            batch_pck = avg_pck_result(output_2D, gt_2D, alpha=0.2)
            total_pck += batch_pck
            count += 1

            all_pred.append(output_2D.detach().cpu())
            all_gt.append(gt_2D.detach().cpu())

        avg_loss = sum(losses) / len(losses)
        avg_pck = total_pck / count

        # ===== 計算每個關節 Smooth L1 誤差 =====
        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        joint_smooth = per_joint_smooth_l1(all_pred, all_gt)  # (17,)

        return avg_loss, avg_pck, joint_smooth



if __name__ == '__main__':
    seed = 1

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'h36m':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, args)
        actions = define_actions(args.actions)

        if args.train:
            train_data = Fusion(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)
    elif args.dataset == '3dhp':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Mpi_inf_3dhp_Dataset(dataset_path, args)
        actions = define_actions_3dhp(args.actions, 0)

        if args.train:
            train_data = Fusion_3dhp(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion_3dhp(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)

    elif args.dataset == 'csi':
        
        def butter_filter(data, cutoff=10, fs=1000, order=4):
            """ 對每個天線資料進行 Butterworth 低通濾波（逐 channel）"""
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
            # data: shape (51, 6, 2025)
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):      # 51 subcarriers
                for j in range(data.shape[1]):  # 6 antennas
                    filtered[i, j] = filtfilt(b, a, data[i, j])
            return filtered
            
        def dwt_denoise(data, wavelet='db4', level=1):
            denoised = np.zeros_like(data)
            for j in range(data.shape[1]):      # 6 antennas
                for k in range(data.shape[2]):  # T subcarriers
                    signal = data[:, j, k]      # 取出 (51,) → 一條時間序列
                    coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)

                    # 計算統計閾值
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

                    # 對細節係數做軟閾值處理
                    coeffs[1:] = [
                        pywt.threshold(c, value=uthresh, mode='soft') 
                        if np.max(np.abs(c)) > 1e-12 else np.zeros_like(c) 
                        for c in coeffs[1:]
                    ]

                    # 重建
                    recon = pywt.waverec(coeffs, wavelet, mode='periodization')
                    denoised[:, j, k] = recon[:signal.shape[0]]
            return denoised


        def normalize(data):
            """ 對每個 channel zero-mean, unit-std """
            mean = np.mean(data, axis=2, keepdims=True)
            std = np.std(data, axis=2, keepdims=True) + 1e-8
            return (data - mean) / std

        class CSI_KPT_Dataset(Dataset):
            def __init__(self, env_roots):
                self.index_map = {}

                for env_path in env_roots:
                    csi_root = os.path.join(env_path, "npy_dwt")
                    kpt_root = os.path.join(env_path, "kpt_npz_vitpose")

                    csi_files = glob.glob(os.path.join(csi_root, "**", "*.npz"), recursive=True)
                    kpt_files = glob.glob(os.path.join(kpt_root, "**", "*.npz"), recursive=True)

                    kpt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in kpt_files}

                    for csi_path in csi_files:
                        timestamp = os.path.splitext(os.path.basename(csi_path))[0]
                        if timestamp in kpt_map:
                            self.index_map[timestamp] = (csi_path, kpt_map[timestamp])

                self.timestamps = sorted(self.index_map.keys())

            def __len__(self):
                return len(self.timestamps)
                

            def __getitem__(self, idx):
                timestamp = self.timestamps[idx]
                csi_path, kpt_path = self.index_map[timestamp]

                # 讀 CSI
                csi_npz = np.load(csi_path)
                mag = csi_npz["mag"].astype(np.float32)
                pha = csi_npz["pha"].astype(np.float32)
                
                # 去噪
                #mag = normalize(butter_filter(mag))
                #pha = normalize(butter_filter(pha))
                # mag = normalize(dwt_denoise(mag))
                # pha = normalize(dwt_denoise(pha))
                '''
                if args.denoise_method == 'dwt':
                    mag = normalize(dwt_denoise(mag))
                    pha = normalize(pha)
                elif args.denoise_method == 'butter':
                    mag = normalize(butter_filter(mag))
                    pha = normalize(pha)
                elif args.denoise_method == 'none':
                    pass  # 不去噪
                '''
                csi = np.concatenate([mag, pha], axis=2)

                # 讀關鍵點
                kpt_npz = np.load(kpt_path)
                keypoints = kpt_npz["keypoints"].astype(np.float32)

                if keypoints.shape[0] > 1:
                    keypoints = keypoints[0]
                elif keypoints.shape == (1, 17, 2):
                    keypoints = keypoints[0]
                elif keypoints.shape != (17, 2):
                    print(f"⚠️ Unexpected shape: {keypoints.shape} at {timestamp}")
                    keypoints = np.zeros((17, 2), dtype=np.float32)

                return torch.tensor(csi), torch.tensor(keypoints)

        # 定義所有 Env 資料夾路徑
        #env_roots = [
        #    os.path.join(args.root_path, f"Env{i}") for i in range(6)
        #]
        env_roots = [os.path.join(args.root_path, "Env1_train")]
        #env_roots = [os.path.join(args.root_path, "Env0")]

        full_dataset = CSI_KPT_Dataset(env_roots)

        # 拆分資料集
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_data, test_data = random_split(full_dataset, [train_size, test_size])

        # DataLoader
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                    num_workers=int(args.workers), pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                    num_workers=int(args.workers), pin_memory=True)

        actions = ['csi']

    model = Model(args).cuda()
    #model = Model(args, svd_path="svd_W_24300_to_512.pt").cuda()
    model_refine = post_refine(args).cuda()
    print(next(model.parameters()).device)

    if args.previous_dir != '':
        Load_model(args, model, model_refine)

    lr = args.lr
    all_param = []
    all_param += list(model.parameters())

    if args.refine:
        all_param += list(model_refine.parameters())

    optimizer = optim.Adam(all_param, lr=lr, amsgrad=True)
    
    ##--------------------------------epoch-------------------------------- ##
    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    joint_weights = None  # 初始化，第一輪不加權

    for epoch in range(1, args.nepoch + 1):
        ## train
        if args.train:
            loss = train(train_dataloader, model, model_refine, optimizer, epoch, joint_weights)
            loss_epochs.append(loss * 1000)

        ## test
        with torch.no_grad():
            if args.dataset == 'csi':
                p1, PCK, joint_smooth = test(actions, test_dataloader, model, model_refine)
                mpjpes.append(p1)

                # 更新 joint_weights
                joint_weights = make_joint_weights(joint_smooth.cuda())
                print(f"Epoch {epoch} Joint SmoothL1: {joint_smooth.cpu().numpy()}")
                print(f"Epoch {epoch} Joint Weights: {joint_weights.cpu().numpy()}")
            else:
                p1, p2, pck, auc = test(actions, test_dataloader, model, model_refine)
                mpjpes.append(p1)

        ## save the best model
        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')

            if args.refine:
                args.previous_refine_name = save_model(args, epoch, p1, model_refine, 'refine')

            args.previous_best = p1

        ## print
        if args.train:
            logging.info('epoch: %d, lr: %.6f, Train loss: %.4f, mpjpe: %.2f, PCK: %.2f' % (epoch, lr, loss, p1, PCK))
            print('%d, lr: %.6f, Train loss: %.4f, RMSE: %.2f, PCK: %.2f' % (epoch, lr, loss, p1, PCK))
            
            
            ## adjust lr
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay 
            '''
            
            # 每 lr_decay_epoch 個 epoch 更新一次學習率
            if epoch % args.lr_decay_epoch == 1 and epoch > 1:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            '''
            
        else:
            if args.dataset == 'h36m':
                print('p1: %.2f, p2: %.2f' % (p1, p2))
            elif args.dataset == '3dhp':
                print('pck: %.2f, auc: %.2f, p1: %.2f, p2: %.2f' % (pck, auc, p1, p2))
            break

        ## training curves
        if epoch == 1:
            start_epoch = 3
                
        if args.train and epoch > start_epoch:
            plt.figure()
            epoch_x = np.arange(start_epoch+1, len(loss_epochs)+1)
            plt.plot(epoch_x, loss_epochs[start_epoch:], '.-', color='C0')
            plt.plot(epoch_x, mpjpes[start_epoch:], '.-', color='C1')
            plt.legend(['Loss', 'Test'])
            plt.ylabel('MPJPE')
            plt.xlabel('Epoch')
            plt.xlim((start_epoch+1, len(loss_epochs)+1))
            plt.savefig(os.path.join(args.checkpoint, 'loss.png'))
            plt.close()
