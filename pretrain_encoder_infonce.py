import os
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pywt


# -----------------------------
# DWT + normalize (跟你一致)
# -----------------------------
def dwt_denoise(data, wavelet='db4', level=2):
    """
    data: [T, A, F] 例如 [51, 6, 2025]
    """
    denoised = np.zeros_like(data, dtype=np.float32)
    T, A, F = data.shape
    for j in range(A):       # antennas
        for k in range(F):   # subcarriers
            signal = data[:, j, k]  # (T,)
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
    """
    data: [T, A, F]
    你原本是沿 subcarriers axis=2 做 mean/std
    """
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True) + 1e-8
    return (data - mean) / std


# -----------------------------
# Dataset: Env*/action/npy/**/*.npz
# label = action
# -----------------------------
class CSI_Action_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        env_names=("Env0_train","Env1_train","Env2_train","Env3_train","Env4_train","Env5_train"),
        use_dwt_folder="npy_dwt",
        per_env_samples=None,  # None=全部；或指定每個 env 抽樣數
        seed=1,
        wavelet="db4",
        dwt_level=2,
    ):
        super().__init__()
        self.samples = []   # list of (csi_npz_path, action_id)
        self.action2id = {}
        self.id2action = {}

        rng = random.Random(seed)

        env_roots = [os.path.join(root_path, e) for e in env_names]
        env_roots = [p for p in env_roots if os.path.isdir(p)]
        if len(env_roots) == 0:
            raise RuntimeError(f"找不到 env 資料夾：{env_names}，root_path={root_path}")

        # 收集：每個 env 的 action 資料
        per_env_lists = []
        for env_path in env_roots:
            env_name = os.path.basename(env_path.rstrip("/"))
            env_list = []

            for action_dir in sorted(os.listdir(env_path)):
                action_path = os.path.join(env_path, action_dir)
                if not os.path.isdir(action_path):
                    continue
                # 過濾非動作資料夾
                if action_dir.startswith(("img_", "kpt_", "npy", "mask", "pose")):
                    continue

                if action_dir not in self.action2id:
                    self.action2id[action_dir] = len(self.action2id)

                csi_root = os.path.join(action_path, use_dwt_folder)
                if not os.path.isdir(csi_root):
                    continue

                files = glob.glob(os.path.join(csi_root, "**", "*.npz"), recursive=True)
                for fp in files:
                    env_list.append((fp, self.action2id[action_dir]))

            if len(env_list) == 0:
                print(f"⚠️ {env_name}: 沒有掃到任何 action npz")
                continue

            # optional sampling per env
            if per_env_samples is not None and per_env_samples > 0 and len(env_list) > per_env_samples:
                env_list = rng.sample(env_list, per_env_samples)
                print(f"✅ {env_name}: sampled {per_env_samples} / (env_total={len(env_list)})")
            else:
                print(f"✅ {env_name}: use {len(env_list)}")

            per_env_lists.append(env_list)

        self.samples = [x for env_list in per_env_lists for x in env_list]
        rng.shuffle(self.samples)

        self.id2action = {v:k for k,v in self.action2id.items()}
        print(f"📌 Total pretrain samples: {len(self.samples)} | actions: {len(self.action2id)}")

        self.wavelet = wavelet
        self.dwt_level = dwt_level

        # 建 class->indices（做 balanced sampling 用）
        self.by_action = defaultdict(list)
        for idx, (_, y) in enumerate(self.samples):
            self.by_action[int(y)].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi_path, y = self.samples[idx]
        npz = np.load(csi_path)
        mag = npz["mag"].astype(np.float32)  # [T,A,F]
        pha = npz["pha"].astype(np.float32)  # [T,A,F]

        # 只對 mag 做去噪
        #mag = normalize(dwt_denoise(mag, wavelet=self.wavelet, level=self.dwt_level))
        #pha = normalize(pha)

        csi = np.concatenate([mag, pha], axis=2)  # [T,A,2F] => 通常 4050
        x = torch.from_numpy(csi).float()
        y = torch.tensor(int(y), dtype=torch.long)
        return x, y


# -----------------------------
# Balanced batch sampler
# batch = n_classes * n_samples
# 確保同 action 至少 2 個樣本 -> 才有正樣本
# -----------------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset: CSI_Action_Dataset, n_classes: int, n_samples: int, batches_per_epoch: int, seed=1):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)

        self.class_ids = list(dataset.by_action.keys())
        if len(self.class_ids) < n_classes:
            raise ValueError(f"動作類別數({len(self.class_ids)}) < n_classes({n_classes})")

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            classes = self.rng.sample(self.class_ids, self.n_classes)
            batch = []
            for c in classes:
                pool = self.dataset.by_action[c]
                if len(pool) >= self.n_samples:
                    idxs = self.rng.sample(pool, self.n_samples)
                else:
                    idxs = [self.rng.choice(pool) for _ in range(self.n_samples)]
                batch.extend(idxs)
            self.rng.shuffle(batch)
            yield batch


# -----------------------------
# Your Subcarrier Cross-Attn (copy from your model)
# -----------------------------
class SubcarrierCrossAttn_Band_Cross(nn.Module):
    def __init__(self, band_size=45, d_model=64, num_heads=4, n_antennas=6):
        super().__init__()
        self.band_size = band_size
        self.d_model = d_model
        self.n_antennas = n_antennas

        self.embed = nn.Linear(band_size, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, band_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, A, F]
        B, T, A, F = x.shape
        assert A == self.n_antennas, f"Expected A={self.n_antennas}, got {A}"
        assert F % self.band_size == 0, f"F={F} must be divisible by band_size={self.band_size}"
        K = F // self.band_size

        x_band = x.view(B, T, A, K, self.band_size)  # [B,T,A,K,band]
        x_band = x_band.permute(0, 1, 3, 2, 4).contiguous().view(B*T*K, A, self.band_size)  # [BTK,A,band]

        h = self.embed(x_band)  # [BTK,A,d]
        outs = []
        for i in range(A):
            q = h[:, i:i+1, :]                               # [BTK,1,d]
            kv = torch.cat([h[:, :i, :], h[:, i+1:, :]], 1)  # [BTK,A-1,d]
            yi, _ = self.attn(q, kv, kv)
            yi = self.norm(yi + q)
            outs.append(yi)

        out = torch.cat(outs, dim=1)  # [BTK,A,d]
        out = self.out(out)           # [BTK,A,band]
        out = out.view(B, T, K, A, self.band_size).permute(0, 1, 3, 2, 4).contiguous().view(B, T, A, F)
        return out


# -----------------------------
# EncoderOnly (pretrain target)
# [B,51,6,4050] -> z:[B,C]
# -----------------------------
class EncoderOnly(nn.Module):
    def __init__(self, channel=256, band_size=45, sc_d_model=64, sc_heads=4, n_antennas=6, feat_dim=4050):
        super().__init__()
        self.n_antennas = n_antennas
        self.feat_dim = feat_dim
        self.channel = channel

        self.subcarrier_attn_band = SubcarrierCrossAttn_Band_Cross(
            band_size=band_size, d_model=sc_d_model, num_heads=sc_heads, n_antennas=n_antennas
        )
        self.embedding = nn.Linear(n_antennas * feat_dim, channel)
        self.temporal_proj = nn.Linear(channel * 2, channel)

    def forward(self, x):
        # x: [B,T,A,F]
        x = self.subcarrier_attn_band(x)
        B, T, A, F = x.shape
        x = x.view(B, T, A * F)
        x = self.embedding(x)  # [B,T,C]

        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        z = torch.cat([mean, std], dim=-1)
        z = self.temporal_proj(z)  # [B,C]
        return z


# -----------------------------
# SupCon / InfoNCE: same label => positive
# -----------------------------
def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature=0.1) -> torch.Tensor:
    B = z.size(0)
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / temperature  # [B,B]

    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    labels = labels.view(-1, 1)
    pos = (labels == labels.t()) & (~eye)

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * (~eye).float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_cnt = pos.sum(dim=1).float()
    mean_log_prob_pos = (pos.float() * log_prob).sum(dim=1) / (pos_cnt + 1e-12)

    valid = (pos_cnt > 0).float()
    loss = -(valid * mean_log_prob_pos).sum() / (valid.sum() + 1e-12)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", type=str, required=True)
    ap.add_argument("--gpu", type=str, default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_classes", type=int, default=8)
    ap.add_argument("--batch_samples", type=int, default=2)
    ap.add_argument("--batches_per_epoch", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_path", type=str, default="checkpoints/encoder_pretrained.pt")

    ap.add_argument("--per_env_samples", type=int, default=10000)  # 你要跟 finetune 一樣每 env 抽 10000 可用
    ap.add_argument("--use_dwt_folder", type=str, default="npy")
    ap.add_argument("--wavelet", type=str, default="db4")
    ap.add_argument("--dwt_level", type=int, default=2)

    # encoder config
    ap.add_argument("--channel", type=int, default=256)
    ap.add_argument("--band_size", type=int, default=45)
    ap.add_argument("--sc_d_model", type=int, default=64)
    ap.add_argument("--sc_heads", type=int, default=4)

    args = ap.parse_args()

    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CSI_Action_Dataset(
        root_path=args.root_path,
        per_env_samples=args.per_env_samples,
        use_dwt_folder=args.use_dwt_folder,
        wavelet=args.wavelet,
        dwt_level=args.dwt_level,
    )

    sampler = BalancedBatchSampler(
        dataset,
        n_classes=args.batch_classes,
        n_samples=args.batch_samples,
        batches_per_epoch=args.batches_per_epoch,
        seed=1,
    )

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    encoder = EncoderOnly(
        channel=args.channel,
        band_size=args.band_size,
        sc_d_model=args.sc_d_model,
        sc_heads=args.sc_heads,
        n_antennas=6,
        feat_dim=4050,
    ).to(device)

    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    encoder.train()
    for ep in range(1, args.epochs + 1):
        total = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            z = encoder(x)
            loss = supcon_loss(z, y, temperature=args.temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(loader)
        print(f"[Pretrain] epoch={ep:03d} loss={avg:.4f}")

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "action2id": dataset.action2id,
                "args": vars(args),
            },
            args.save_path
        )
        print(f"  saved -> {args.save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
