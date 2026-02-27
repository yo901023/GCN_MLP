import os
import glob
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pywt

# 你原本的 Model（含 GCN）：
from model.graphmlp_transformer_cross import Model  # 確保路徑跟你專案一致


# -----------------------------
# DWT + normalize (跟你一致)
# -----------------------------
def dwt_denoise(data, wavelet='db4', level=2):
    denoised = np.zeros_like(data, dtype=np.float32)
    T, A, F = data.shape
    for j in range(A):
        for k in range(F):
            signal = data[:, j, k]
            coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if np.max(np.abs(coeffs[-1])) > 1e-12 else 0.0
            uthresh = sigma * np.sqrt(2 * np.log(len(signal) + 1e-12))

            new_coeffs = [coeffs[0]]
            for c in coeffs[1:]:
                if np.max(np.abs(c)) > 1e-12:
                    new_coeffs.append(pywt.threshold(c, value=uthresh, mode='soft'))
                else:
                    new_coeffs.append(np.zeros_like(c))
            recon = pywt.waverec(new_coeffs, wavelet, mode='periodization')
            denoised[:, j, k] = recon[:T]
    return denoised


def normalize(data):
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True) + 1e-8
    return (data - mean) / std


# -----------------------------
# Weighted SmoothL1 (跟你一致)
# -----------------------------
def weighted_smooth_l1_loss(predicted, target, joint_weights, beta=1.0):
    mask = (target.abs().sum(dim=-1) > 0).float()     # (B,J)
    diff = (predicted - target).abs()                 # (B,J,2)

    smooth = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    ).mean(dim=-1)  # (B,J)

    weighted = smooth * joint_weights[None, :] * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom


def per_joint_smooth_l1(predicted, target, beta=1.0):
    mask = (target.abs().sum(dim=-1) > 0)  # (N,J)
    diff = (predicted - target).abs()      # (N,J,2)
    smooth = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    ).mean(dim=-1)  # (N,J)

    J = predicted.shape[1]
    out = torch.zeros(J, device=predicted.device)
    for j in range(J):
        valid = mask[:, j]
        out[j] = smooth[valid, j].mean() if valid.sum() > 0 else 0.0
    return out


def make_joint_weights(smooth_err, beta=1.0, eps=1e-6):
    normed = (smooth_err + eps) / (smooth_err.mean() + eps)
    w = normed ** beta
    w = w / w.mean()
    w = w.clamp(min=0.5, max=2.0)
    return w


def avg_pck_result(predicted, target, alpha=0.2, left_shoulder=6, right_hip=11):
    B, J, _ = predicted.shape
    ls_valid = (target[:, left_shoulder].abs().sum(dim=-1) > 0)
    rh_valid = (target[:, right_hip].abs().sum(dim=-1) > 0)
    sample_valid = ls_valid & rh_valid

    shoulder = torch.norm(target[:, left_shoulder] - target[:, right_hip], dim=-1).clamp(min=1e-6)
    dists = torch.norm(predicted - target, dim=-1)
    normalized = dists / shoulder[:, None]

    joint_valid = (target.abs().sum(dim=-1) > 0) & sample_valid[:, None]
    if joint_valid.sum() == 0:
        return 0.0
    correct = (normalized <= alpha) & joint_valid
    return (correct.float().sum() / joint_valid.float().sum()).item()


# -----------------------------
# CSI-KPT dataset (你原本邏輯)
# -----------------------------
class CSI_KPT_Dataset(Dataset):
    def __init__(self, env_roots, per_env_samples=10000, seed=1, use_dwt_folder="npy", kpt_folder="kpt_npz_vitpose",
                 wavelet="db4", dwt_level=2):
        self.pairs = []
        rng = random.Random(seed)

        self.wavelet = wavelet
        self.dwt_level = dwt_level
        self.use_dwt_folder = use_dwt_folder
        self.kpt_folder = kpt_folder

        for env_path in env_roots:
            env_name = os.path.basename(env_path.rstrip("/"))
            csi_root = os.path.join(env_path, use_dwt_folder)
            kpt_root = os.path.join(env_path, kpt_folder)

            csi_files = glob.glob(os.path.join(csi_root, "**", "*.npz"), recursive=True)
            kpt_files = glob.glob(os.path.join(kpt_root, "**", "*.npz"), recursive=True)
            kpt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in kpt_files}

            env_pairs = []
            for csi_path in csi_files:
                ts = os.path.splitext(os.path.basename(csi_path))[0]
                if ts in kpt_map:
                    env_pairs.append((csi_path, kpt_map[ts]))

            if len(env_pairs) == 0:
                print(f"⚠️ {env_name}: no matched pairs")
                continue

            if per_env_samples is not None and len(env_pairs) > per_env_samples:
                env_pairs = rng.sample(env_pairs, per_env_samples)
                print(f"✅ {env_name}: sampled {per_env_samples} (matched {len(env_pairs)})")
            else:
                print(f"✅ {env_name}: use {len(env_pairs)} matched")

            self.pairs.extend(env_pairs)

        rng.shuffle(self.pairs)
        print(f"📌 Total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        csi_path, kpt_path = self.pairs[idx]

        csi_npz = np.load(csi_path)
        mag = csi_npz["mag"].astype(np.float32)
        pha = csi_npz["pha"].astype(np.float32)

        #mag = normalize(dwt_denoise(mag, wavelet=self.wavelet, level=self.dwt_level))
        #pha = normalize(pha)
        csi = np.concatenate([mag, pha], axis=2)  # [T,A,4050]

        kpt_npz = np.load(kpt_path)
        keypoints = kpt_npz["keypoints"].astype(np.float32)

        # 你原本的處理
        if keypoints.shape[0] > 1:
            keypoints = keypoints[0]
        elif keypoints.shape == (1, 17, 2):
            keypoints = keypoints[0]
        elif keypoints.shape != (17, 2):
            keypoints = np.zeros((17, 2), dtype=np.float32)

        return torch.tensor(csi), torch.tensor(keypoints)


# -----------------------------
# load pretrained encoder into full model + freeze
# -----------------------------
def load_pretrained_encoder_into_model(model: nn.Module, ckpt_path: str, freeze_encoder: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_state = ckpt["encoder"] if "encoder" in ckpt else ckpt  # 支援你直接存 state_dict 的情況

    # 只把 encoder 對應到 model 裡同名參數（strict=False）
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    print("✅ Loaded pretrained weights (strict=False)")
    if missing:
        print(f"  missing keys: {len(missing)} (正常，因為 full model 还包含 GCN/head)")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")

    if freeze_encoder:
        # 依你 model 命名：subcarrier_attn_band / embedding / temporal_proj
        for name, p in model.named_parameters():
            if ("subcarrier_attn_band" in name) or ("embedding" in name) or ("temporal_proj" in name):
                p.requires_grad = False
        print("✅ Encoder frozen (subcarrier_attn_band/embedding/temporal_proj)")


def train_one_epoch(model, loader, optimizer, joint_weights=None, device="cuda"):
    model.train()
    criterion = nn.SmoothL1Loss(beta=1.0)

    total_loss = 0.0
    for x, gt in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        gt = gt.to(device)

        pred = model(x)

        if joint_weights is None:
            loss = criterion(pred, gt)
        else:
            loss = weighted_smooth_l1_loss(pred, gt, joint_weights.to(device), beta=1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device="cuda"):
    model.eval()
    losses = []
    pcks = []
    all_pred, all_gt = [], []

    for x, gt in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        gt = gt.to(device)

        pred = model(x)
        # 你原本是用 mpjpe；這裡簡化用 L2 mean 當 proxy（你要換回 eval_cal.mpjpe 也行）
        loss = torch.norm(pred - gt, dim=-1).mean()
        losses.append(loss.item())

        pcks.append(avg_pck_result(pred, gt, alpha=0.2))

        all_pred.append(pred.detach().cpu())
        all_gt.append(gt.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    joint_smooth = per_joint_smooth_l1(all_pred, all_gt)

    return float(np.mean(losses)), float(np.mean(pcks)), joint_smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", type=str, required=True)
    ap.add_argument("--gpu", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--per_env_samples", type=int, default=10000)
    ap.add_argument("--use_dwt_folder", type=str, default="npy")
    ap.add_argument("--kpt_folder", type=str, default="kpt_npz_vitpose")
    ap.add_argument("--wavelet", type=str, default="db4")
    ap.add_argument("--dwt_level", type=int, default=2)

    ap.add_argument("--load_pretrained_encoder", type=str, default="")
    ap.add_argument("--freeze_encoder", type=int, default=1)
    ap.add_argument("--save_path", type=str, default="checkpoints/pose_finetuned.pt")

    # 這些要跟你的 Model(args) 一致
    ap.add_argument("--channel", type=int, default=256)
    ap.add_argument("--frames", type=int, default=51)
    ap.add_argument("--n_joints", type=int, default=17)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d_hid", type=int, default=256)
    ap.add_argument("--token_dim", type=int, default=64)
    ap.add_argument("--out_channels", type=int, default=2)

    args = ap.parse_args()

    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_roots = [os.path.join(args.root_path, f"Env{i}_train") for i in range(6)]
    dataset = CSI_KPT_Dataset(
        env_roots,
        per_env_samples=args.per_env_samples,
        seed=1,
        use_dwt_folder=args.use_dwt_folder,
        kpt_folder=args.kpt_folder,
        wavelet=args.wavelet,
        dwt_level=args.dwt_level,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # 用 argparse args 直接餵給你原本的 Model
    model = Model(args).to(device)

    if args.load_pretrained_encoder:
        load_pretrained_encoder_into_model(
            model, args.load_pretrained_encoder, freeze_encoder=bool(args.freeze_encoder)
        )

    # 只訓練 requires_grad=True 的參數
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, amsgrad=True)

    best = 1e9
    joint_weights = None

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, joint_weights=joint_weights, device=device)
        test_loss, test_pck, joint_smooth = eval_one_epoch(model, test_loader, device=device)

        # 更新 joint weights（你原本做法）
        joint_weights = make_joint_weights(joint_smooth.to(device))

        print(f"Epoch {ep:03d} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | PCK={test_pck:.4f}")
        print(f"  joint_smooth={joint_smooth.numpy()}")
        print(f"  joint_weights={joint_weights.detach().cpu().numpy()}")

        if test_loss < best:
            best = test_loss
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "best": best,
                    "args": vars(args),
                },
                args.save_path
            )
            print(f"✅ saved best -> {args.save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
