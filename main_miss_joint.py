#GT沒有點座標就不計算
#使用DwT、embedding用線性層、把graph改成coco(已經考慮對稱的結構)、GCN鄰接矩陣用yolo
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
#from model.graphmlp import Model
#from model.graphmlp_attention import Model
from model.graphmlp_origin import Model
from scipy.signal import butter, filtfilt
import pywt

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def masked_mse_loss(predicted, target):
    """
    針對 2D 關節 (B,J,2)，忽略 target 為 (0,0) 的關節，做 MSE 平均。
    """
    # mask: [B,J]，True 代表該關節有標註
    mask = (target.abs().sum(dim=-1) > 0).float()          # 1=valid, 0=ignore
    # 每個關節的 MSE（x,y 取 mean，避免和關節數/批次混在一起）
    per_joint_mse = ((predicted - target) ** 2).mean(dim=-1)   # [B,J]
    # 只有有效的關節參與平均
    denom = mask.sum().clamp(min=1.0)
    return (per_joint_mse * mask).sum() / denom


def masked_mpjpe(predicted, target):
    """
    針對 2D 關節 (B,J,2)，忽略 target 為 (0,0) 的關節，計算 MPJPE（實際上是 2D 平面上的平均歐式距離）。
    """
    mask = (target.abs().sum(dim=-1) > 0)                  # [B,J] bool
    dists = torch.norm(predicted - target, dim=-1)         # [B,J]
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predicted.device)
    return dists[mask].mean()


def avg_pck_result(predicted, target, alpha=0.2, left_shoulder=6, right_shoulder=11):
    """
    PCK@alpha，忽略 target 為 (0,0) 的關節；若肩點缺失，該樣本不計。
    """
    assert predicted.shape == target.shape
    B, J, _ = predicted.shape

    # 樣本是否肩關節皆有效
    ls_valid = (target[:, left_shoulder].abs().sum(dim=-1) > 0)
    rs_valid = (target[:, right_shoulder].abs().sum(dim=-1) > 0)
    sample_valid = ls_valid & rs_valid                     # [B]

    # 肩寬（避免除 0）
    shoulder_dists = torch.norm(
        target[:, left_shoulder] - target[:, right_shoulder], dim=-1
    ).clamp(min=1e-6)                                      # [B]

    dists = torch.norm(predicted - target, dim=-1)         # [B,J]
    normalized = dists / shoulder_dists[:, None]           # [B,J]

    # 關節有效（非 (0,0)）且樣本肩寬有效
    joint_valid = (target.abs().sum(dim=-1) > 0) & sample_valid[:, None]  # [B,J]
    if joint_valid.sum() == 0:
        return 0.0

    correct = (normalized <= alpha) & joint_valid
    return (correct.float().sum() / joint_valid.float().sum()).item()

def train(dataloader, model, model_refine, optimizer, epoch):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader, 0)):
        if args.dataset == 'csi':
            input_2D, gt_2D = data
            input_2D = input_2D.cuda()
            gt_2D = gt_2D.cuda()
            output_2D = model(input_2D)
            #loss = nn.MSELoss()(output_2D, gt_2D)
            loss = masked_mse_loss(output_2D, gt_2D)   # 忽略 (0,0) 關節
            
            

        else:
            batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
            [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('train', [input_2D, input_2D_GT, gt_3D, batch_cam])

            output_3D = model(input_2D)

            out_target = gt_3D.clone()
            out_target[:, :, args.root_joint] = 0
            out_target = out_target[:, args.pad].unsqueeze(1)

            if args.refine:
                model_refine.train()
                output_3D = refine_model(model_refine, output_3D, input_2D, gt_3D, batch_cam, args.pad, args.root_joint)
                loss = eval_cal.mpjpe(output_3D, out_target)
            else:
                loss = eval_cal.mpjpe(output_3D, out_target)

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
        for data in tqdm(dataloader, 0):
            input_2D, gt_2D = data
            input_2D = input_2D.cuda()
            gt_2D = gt_2D.cuda()
            output_2D = model(input_2D)
            #loss = eval_cal.mpjpe(output_2D, gt_2D)
            loss = masked_mpjpe(output_2D, gt_2D)   # 忽略 (0,0) 關節
            losses.append(loss.item())
            # 計算 PCK@0.2
            #batch_pck = avg_pck_result(output_2D.detach().cpu(), gt_2D.detach().cpu(), alpha=0.2)
            batch_pck = avg_pck_result(output_2D, gt_2D, alpha=0.2)
            total_pck += batch_pck
            count += 1

        avg_loss = sum(losses) / len(losses)
        avg_pck = total_pck / count
        return avg_loss, avg_pck

    action_error = define_error_list(actions)
    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, gt_3D, batch_cam])

        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, args.joints_left + args.joints_right, :] = output_3D_flip[:, :, args.joints_right + args.joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        out_target = out_target[:, args.pad].unsqueeze(1)

        if args.refine:
            model_refine.eval()
            output_3D = refine_model(model_refine, output_3D, input_2D[:, 0], gt_3D, batch_cam, args.pad, args.root_joint)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0

        action_error = eval_cal.test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)

    p1, p2, pck, auc = print_error(args.dataset, action_error, args.train)
    return p1, p2, pck, auc

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
            for i in range(data.shape[0]):        # 51 subcarriers
                for j in range(data.shape[1]):    # 6 antennas
                    signal = data[i, j]
                    coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)
                    
                    # 計算統計閾值
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
                    
                    # 對細節係數做軟閾值處理（不是直接歸零）
                    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
                    
                    recon = pywt.waverec(coeffs, wavelet, mode='periodization')
                    denoised[i, j] = recon[:signal.shape[0]]
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
                    csi_root = os.path.join(env_path, "npy")
                    kpt_root = os.path.join(env_path, "kpt_npz")

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

                if args.denoise_method == 'dwt':
                    mag = normalize(dwt_denoise(mag))
                    pha = normalize(dwt_denoise(pha))
                elif args.denoise_method == 'butter':
                    mag = normalize(butter_filter(mag))
                    pha = normalize(butter_filter(pha))
                elif args.denoise_method == 'none':
                    pass  # 不去噪
                    
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
        env_roots = [os.path.join(args.root_path, "Env0")]

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

    for epoch in range(1, args.nepoch + 1):
        ## train
        if args.train: 
            loss = train(train_dataloader, model, model_refine, optimizer, epoch)
            loss_epochs.append(loss * 1000)

        ## test
        with torch.no_grad():
            #p1, p2, pck, auc = test(actions, test_dataloader, model, model_refine)
            p1, PCK= test(actions, test_dataloader, model, model_refine)
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
            print('%d, lr: %.6f, Train loss: %.4f, mpjpe: %.2f, PCK: %.2f' % (epoch, lr, loss, p1, PCK))
            
            
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
