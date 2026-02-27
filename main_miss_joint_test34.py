#GT沒有點座標就不計算、只對振幅做去噪
#使用DwT、embedding用線性層、把graph改成coco(已經考慮對稱的結構)、GCN鄰接矩陣用yolo
#針對不同部分給予不同權重(test3是用RMSE倒數，test5是用RMSE)、DwT改成沿著時間做去噪
#調整權重beta值(beta=1)、把training set 裡面看起來與人體偏差很多的資料拿掉
#loss function改成smooth L1、權重也用成smooth L1
#51維轉17維:51把維轉成1維，再複製成17維丟進mlpgcn
#用mmfi訓練
#做transformer(對天線做cross attention、45個為一組)
import os
import re
import glob
import torch
import csv, json
import random
import scipy.io as sio
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
#from model.graphmlp_origin import Model
#from model.graphmlp_pca import Model
from model.graphmlp_transformer_cross_DTPose import Model
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
    
def pck_multi(predicted, target, alphas=(0.1,0.2,0.3,0.4,0.5), left_shoulder=6, right_hip=11):
    """
    回傳每個 alpha 的 PCK；忽略 target 為 (0,0) 的關節；肩點缺失樣本不計。
    predicted, target: (B,J,2)
    """
    assert predicted.shape == target.shape
    B, J, _ = predicted.shape

    ls_valid = (target[:, left_shoulder].abs().sum(dim=-1) > 0)
    rs_valid = (target[:, right_hip].abs().sum(dim=-1) > 0)
    sample_valid = ls_valid & rs_valid

    shoulder_dists = torch.norm(
        target[:, left_shoulder] - target[:, right_hip], dim=-1
    ).clamp(min=1e-6)  # (B,)

    dists = torch.norm(predicted - target, dim=-1)          # (B,J)
    normalized = dists / shoulder_dists[:, None]            # (B,J)

    joint_valid = (target.abs().sum(dim=-1) > 0) & sample_valid[:, None]  # (B,J)
    denom = joint_valid.float().sum().item()
    if denom == 0:
        return {a: 0.0 for a in alphas}

    out = {}
    for a in alphas:
        correct = (normalized <= a) & joint_valid
        out[a] = (correct.float().sum().item() / denom)
    return out

def train(dataloader, model, model_refine, optimizer, epoch, joint_weights=None):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader, 0)):
        if args.dataset == 'csi':
            input_2D, gt_2D = data
            input_2D = input_2D.cuda()
            gt_2D = gt_2D.cuda()
            # ---- before forward ----
            if torch.isnan(input_2D).any() or torch.isinf(input_2D).any():
                print("❌ NaN/Inf in input_2D")
                print("min/max:", input_2D.min().item(), input_2D.max().item())
                raise SystemExit

            if torch.isnan(gt_2D).any() or torch.isinf(gt_2D).any():
                print("❌ NaN/Inf in gt_2D")
                print("min/max:", gt_2D.min().item(), gt_2D.max().item())
                raise SystemExit
            output_2D = model(input_2D)
            # ---- after forward ----
            if torch.isnan(output_2D).any() or torch.isinf(output_2D).any():
                print("❌ NaN/Inf in output_2D")
                raise SystemExit
                
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


def test(actions, dataloader, model, model_refine, alphas=(0.1,0.2,0.3,0.4,0.5)):
    model.eval()

    if args.dataset == 'csi':
        count = 0
        losses = []
        all_pred, all_gt = [], []

        # 累積每個 alpha
        pck_sum = {a: 0.0 for a in alphas}

        for data in tqdm(dataloader, 0):
            input_2D, gt_2D = data
            input_2D = input_2D.cuda()
            gt_2D = gt_2D.cuda()

            output_2D = model(input_2D)

            loss = eval_cal.mpjpe(output_2D, gt_2D)
            losses.append(loss.item())

            # 這裡一次算多個 PCK
            pck_dict = pck_multi(output_2D, gt_2D, alphas=alphas)
            for a in alphas:
                pck_sum[a] += pck_dict[a]

            count += 1
            all_pred.append(output_2D.detach().cpu())
            all_gt.append(gt_2D.detach().cpu())

        avg_loss = sum(losses) / len(losses)

        # 每個 alpha 的平均 PCK（對 batch 取平均）
        avg_pcks = {a: (pck_sum[a] / count) for a in alphas}

        # 計算每個關節 Smooth L1 誤差
        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        joint_smooth = per_joint_smooth_l1(all_pred, all_gt)  # (17,)

        return avg_loss, avg_pcks, joint_smooth

def save_last_ckpt(args, epoch, model, model_refine, optimizer, joint_weights):
    os.makedirs(args.checkpoint, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "refine": model_refine.state_dict(),
        "optimizer": optimizer.state_dict(),
        "previous_best": args.previous_best,
        "joint_weights": joint_weights.detach().cpu() if joint_weights is not None else None,
    }
    torch.save(ckpt, os.path.join(args.checkpoint, "last_ckpt.pt"))


def load_last_ckpt_if_exists(args, model, model_refine, optimizer, device="cuda"):
    ckpt_path = os.path.join(args.checkpoint, "last_ckpt.pt")
    if not os.path.exists(ckpt_path):
        return 1, None  # start_epoch, joint_weights

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model_refine.load_state_dict(ckpt["refine"])
    optimizer.load_state_dict(ckpt["optimizer"])

    args.previous_best = ckpt.get("previous_best", args.previous_best)

    jw = ckpt.get("joint_weights", None)
    joint_weights = jw.to(device) if jw is not None else None

    start_epoch = ckpt["epoch"] + 1
    print(f"✅ Resumed from {ckpt_path}, start_epoch={start_epoch}, previous_best={args.previous_best}")
    return start_epoch, joint_weights

alphas = (0.1, 0.2, 0.3, 0.4, 0.5)

# 每個 alpha 的最佳 PCK 與最佳 epoch
best_pck = {a: -1.0 for a in alphas}
best_pck_epoch = {a: -1 for a in alphas}

# 每個 epoch 的紀錄（用來存檔）
pck_history = []  # list of dict

def save_pck_history(checkpoint_dir, pck_history):
    """
    儲存每個 epoch 的 rmse + PCK@0.1~0.5
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if len(pck_history) == 0:
        return

    # csv
    csv_path = os.path.join(checkpoint_dir, "pck_history.csv")
    fieldnames = list(pck_history[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pck_history)

    # npy
    np.save(os.path.join(checkpoint_dir, "pck_history.npy"), pck_history)


def save_best_pck(checkpoint_dir, best_pck, best_pck_epoch, alphas):
    """
    儲存 PCK@0.1~0.5 各自的最佳值 + epoch
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_dict = {
        "pck_best": {f"pck@{a:.1f}": float(best_pck[a]) for a in alphas},
        "pck_best_epoch": {f"pck@{a:.1f}": int(best_pck_epoch[a]) for a in alphas},
    }

    # json
    json_path = os.path.join(checkpoint_dir, "best_pck.json")
    with open(json_path, "w") as f:
        json.dump(best_dict, f, indent=2)

    # csv
    csv_path = os.path.join(checkpoint_dir, "best_pck.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "best_pck", "best_epoch"])
        for a in alphas:
            writer.writerow([f"pck@{a:.1f}", float(best_pck[a]), int(best_pck_epoch[a])])

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
            
        def dwt_denoise_time(data, wavelet="haar", level=1, mode="periodization", eps=1e-6):
            """
            data: (T,A,F)
            對「不是常數」的時間訊號才做 DWT
            """
            T, A, F = data.shape
            out = np.zeros_like(data, dtype=np.float32)

            w = pywt.Wavelet(wavelet)
            max_level = pywt.dwt_max_level(T, w.dec_len)
            use_level = min(level, max_level)

            # T 太短，直接不做
            if use_level < 1:
                return data.astype(np.float32)

            for a in range(A):
                for f in range(F):
                    sig = data[:, a, f]

                    # 🔑 關鍵：近乎常數 → 直接回傳原訊號
                    if np.std(sig) < eps:
                        out[:, a, f] = sig
                        continue

                    coeffs = pywt.wavedec(sig, wavelet, mode=mode, level=use_level)

                    # detail coeffs 才 threshold
                    for i in range(1, len(coeffs)):
                        c = coeffs[i]
                        mag = np.abs(c)

                        # 避免除 0
                        mask = mag > eps
                        c_new = np.zeros_like(c)
                        c_new[mask] = np.sign(c[mask]) * np.maximum(mag[mask] - eps, 0)
                        coeffs[i] = c_new

                    recon = pywt.waverec(coeffs, wavelet, mode=mode)
                    out[:, a, f] = recon[:T]

            return out


        def normalize(data):
            """ 對每個 channel zero-mean, unit-std """
            mean = np.mean(data, axis=2, keepdims=True)
            std = np.std(data, axis=2, keepdims=True) + 1e-8
            return (data - mean) / std

        def _pick_mat_tensor(mat_dict):
            """
            從 .mat 裡挑出真正的 CSI tensor
            - 優先找常見 key
            - 找不到就選第一個 3D ndarray
            """
            # 常見 key（你也可以自己加）
            for k in ["csi", "CSI", "wifi_csi", "data", "tensor", "feat", "feature"]:
                if k in mat_dict and isinstance(mat_dict[k], np.ndarray):
                    return mat_dict[k]

            # fallback：挑第一個 3D ndarray
            for k, v in mat_dict.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    return v

            raise ValueError(f"Cannot find 3D tensor in mat keys={list(mat_dict.keys())}")


        def _to_TAF(x, A=3, F=114, T=10):
            """
            把輸入 tensor 轉成 [T, A, F]
            mmfi CSI 你說是 3*114*10，但有些存法可能是 10*3*114 或 3*10*114
            這裡自動判斷並轉置
            """
            x = np.asarray(x)

            if x.shape == (A, F, T):
                x = np.transpose(x, (2, 0, 1))  # (T, A, F)
            elif x.shape == (A, T, F):
                x = np.transpose(x, (1, 0, 2))  # (T, A, F)
            elif x.shape == (T, A, F):
                pass
            elif x.shape == (T, F, A):
                x = np.transpose(x, (0, 2, 1))  # (T, A, F)
            elif x.shape == (F, A, T):
                x = np.transpose(x, (2, 1, 0))  # (T, A, F)
            elif x.shape == (F, T, A):
                x = np.transpose(x, (1, 2, 0))  # (T, A, F)
            else:
                raise ValueError(f"Unexpected CSI shape: {x.shape}, cannot map to (T,A,F)=({T},{A},{F})")

            return x.astype(np.float32)



        class MMFI_CSI_GT_Dataset(Dataset):
            """
            MMFi:
            CSI: .../wifi-csi/frameXXX.mat  keys: CSIamp, CSIphase, shape (3,114,10)
            GT : .../rgb/frameXXX.npy       shape (17,2)

            return:
            csi: (T,A,2F) = (10,3,228)   # mag+pha concat on last dim
            gt : (17,2)
            """
            def __init__(self, seq_roots):
                self.pairs = []

                for root in seq_roots:
                    wifi_dir = os.path.join(root, "wifi-csi")
                    gt_dir   = os.path.join(root, "rgb")

                    wifi_files = sorted(glob.glob(os.path.join(wifi_dir, "frame*.mat")))
                    gt_files   = sorted(glob.glob(os.path.join(gt_dir,   "frame*.npy")))

                    def fid(p):
                        # frame001.mat / frame001.npy -> frame001
                        return os.path.splitext(os.path.basename(p))[0]

                    gt_map = {fid(p): p for p in gt_files}

                    for w in wifi_files:
                        k = fid(w)
                        if k in gt_map:
                            self.pairs.append((w, gt_map[k]))

                self.pairs.sort()
                print(f"MMFI pairs: {len(self.pairs)}")

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                wifi_path, gt_path = self.pairs[idx]

                # ---- CSI ----
                mat = sio.loadmat(wifi_path)
                mag = mat["CSIamp"].astype(np.float32)     # (3,114,10)
                pha = mat["CSIphase"].astype(np.float32)   # (3,114,10)

                # (A,F,T) -> (T,A,F)
                mag = np.transpose(mag, (2, 0, 1))  # (10,3,114)
                pha = np.transpose(pha, (2, 0, 1))  # (10,3,114)
                mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
                pha = np.nan_to_num(pha, nan=0.0, posinf=0.0, neginf=0.0)               
                mag = dwt_denoise_time(mag)

                mag = normalize(mag)
                pha = normalize(pha)

                # 合併 (10,3,228)
                csi = np.concatenate([mag, pha], axis=2).astype(np.float32)

                # ---- GT ----
                gt = np.load(gt_path).astype(np.float32)  # (17,2)
                if gt.shape != (17, 2):
                    print(f"⚠️ Unexpected GT shape {gt.shape} at {gt_path}")
                    gt = np.zeros((17, 2), dtype=np.float32)

                return torch.tensor(csi), torch.tensor(gt)


        # 定義所有 Env 資料夾路徑
        mmfi_base = "/media/main/HDD/yo/DT-Pose-main/data/mmfi/dataset"
        E_folders = ["E01", "E02", "E03", "E04"]

        mmfi_seq_roots = []
        for E in E_folders:
            mmfi_seq_roots += sorted(glob.glob(os.path.join(mmfi_base, E, "S*", "A*")))

        print("Total seq_roots:", len(mmfi_seq_roots))
        full_dataset = MMFI_CSI_GT_Dataset(mmfi_seq_roots)


        # 拆分資料集
        train_size = int(0.75 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_data, test_data = random_split(full_dataset, [train_size, test_size])

        # DataLoader
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                    num_workers=int(args.workers), pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                    num_workers=int(args.workers), pin_memory=True)

        actions = ['csi']

    model = Model(args).cuda()
    #model = Model(args, pca_path="pca_24300_to_512.npz").cuda()
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
    start_epoch, joint_weights = load_last_ckpt_if_exists(
        args, model, model_refine, optimizer, device="cuda"
    )
    
    ##--------------------------------epoch-------------------------------- ##
    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    joint_weights = None  # 初始化，第一輪不加權

    for epoch in range(start_epoch, args.nepoch + 1):
        ## train
        if args.train:
            save_last_ckpt(args, epoch, model, model_refine, optimizer, joint_weights)
            loss = train(train_dataloader, model, model_refine, optimizer, epoch, joint_weights)
            loss_epochs.append(loss * 1000)

        with torch.no_grad():
            if args.dataset == 'csi':
                p1, pck_dict, joint_smooth = test(
                    actions, test_dataloader, model, model_refine, alphas=alphas
                )
                mpjpes.append(p1)

                # 更新 joint_weights
                joint_weights = make_joint_weights(joint_smooth.cuda())

                # ---- 印出本 epoch 的 PCK ----
                pck_str = " ".join([f"PCK@{a:.1f}={pck_dict[a]*100:.2f}" for a in alphas])
                print(f"Epoch {epoch} {pck_str}")

                # ---- 更新各 alpha 的最佳值 ----
                for a in alphas:
                    if pck_dict[a] > best_pck[a]:
                        best_pck[a] = pck_dict[a]
                        best_pck_epoch[a] = epoch

                # ---- 印出目前最佳 ----
                best_str = " ".join([
                    f"best PCK@{a:.1f}={best_pck[a]*100:.2f} (epoch {best_pck_epoch[a]})"
                    for a in alphas
                ])
                print(">>", best_str)

                row = {"epoch": int(epoch), "rmse": float(p1)}
                for a in alphas:
                    row[f"pck@{a:.1f}"] = float(pck_dict[a])
                pck_history.append(row)

                # ✅ 每個 epoch 都存：history（csv + npy）
                save_pck_history(args.checkpoint, pck_history)

                # ✅ 每個 epoch 都存：best（json + csv）
                save_best_pck(args.checkpoint, best_pck, best_pck_epoch, alphas)


        ## save the best model
        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')

            if args.refine:
                args.previous_refine_name = save_model(args, epoch, p1, model_refine, 'refine')

            args.previous_best = p1

        ## print
        if args.train:
            #logging.info('epoch: %d, lr: %.6f, Train loss: %.4f, mpjpe: %.2f, PCK: %.2f' % (epoch, lr, loss, p1, pck))
            pck_str2 = " ".join([f"{pck_dict[a]*100:.2f}" for a in alphas])
            print('%d, lr: %.6f, Train loss: %.4f, RMSE: %.2f, %s' %
                (epoch, lr, loss, p1, pck_str))
            
            
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
