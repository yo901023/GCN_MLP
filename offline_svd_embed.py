# offline_svd_embed.py
import os
import glob
import argparse
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
import pywt
from sklearn.decomposition import TruncatedSVD


# ===========================
# 1. 跟原訓練程式一樣的前處理
# ===========================
def butter_filter(data, cutoff=10, fs=1000, order=4):
    """ 對每個天線資料進行 Butterworth 低通濾波（逐 channel）"""
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    # data: shape (51, 6, T)
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):      # 51 time
        for j in range(data.shape[1]):  # 6 antennas
            filtered[i, j] = filtfilt(b, a, data[i, j])
    return filtered


def dwt_denoise(data, wavelet='db4', level=1):
    denoised = np.zeros_like(data)
    # data: [T, A, F]
    for j in range(data.shape[1]):      # 6 antennas
        for k in range(data.shape[2]):  # F subcarriers
            signal = data[:, j, k]      # (T,) 一條時間序列
            coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)

            # 統計閾值
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

            # 對細節係數做軟閾值
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
    """ 對每個 channel zero-mean, unit-std （沿時間維做）"""
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True) + 1e-8
    return (data - mean) / std


# ===========================
# 2. CSI_KPT_Dataset：直接搬你的版本
# ===========================
class CSI_KPT_Dataset(Dataset):
    def __init__(self, env_roots):
        self.index_map = {}

        for env_path in env_roots:
            csi_root = os.path.join(env_path, "npy_dwt")
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

        # ====== 這裡跟你訓練程式一樣：目前只 normalize，不做 DWT / Butter ======
        # mag = normalize(butter_filter(mag))
        # pha = normalize(butter_filter(pha))
        # mag = normalize(dwt_denoise(mag))
        # pha = normalize(dwt_denoise(pha))

        mag = normalize(mag)
        pha = normalize(pha)
        csi = np.concatenate([mag, pha], axis=2)   # shape: [T, A, F'] or [51, 6, 4050]

        # 讀關鍵點（雖然 SVD 不用，但 Dataset 介面保留）
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


# ===========================
# 3. 建 DataLoader + 抽樣資料做 SVD
# ===========================
def build_csi_loader(root_path, batch_size=8, workers=4):
    """
    這裡照你訓練程式：
    env_roots = [os.path.join(args.root_path, "Env0_ViTPose_huge_train")]
    """
    env_roots = [os.path.join(root_path, "Env0_ViTPose_huge_train")]
    dataset = CSI_KPT_Dataset(env_roots)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )
    return loader, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
                        help="資料集根目錄，例如 /media/main/HDD/yo/dataset")
    parser.add_argument('--out_dim', type=int, default=512,
                        help="SVD 降維後的維度 (要跟 model.args.channel 一致)")
    parser.add_argument('--max_samples', type=int, default=5000,
                        help="最多抽多少個 time step 來做 SVD")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='svd_W_24300_to_512.pt')
    args = parser.parse_args()

    # 固定 random seed（可有可無）
    torch.manual_seed(1)
    np.random.seed(1)

    # 1. 建立 DataLoader（只需要 train 的資料就夠了）
    loader, dataset = build_csi_loader(args.root_path, args.batch_size, args.workers)
    print(f"Dataset size: {len(dataset)} samples")

    # 2. 抽樣 CSI，reshape 成 [N, 24300] 做 SVD
    X_list = []
    total = 0
    in_dim = None

    for csi, _ in loader:   # csi: [B, T, A, F]
        # 轉成 float32 + CPU
        csi = csi.float()
        B, T, A, F = csi.shape
        if in_dim is None:
            in_dim = A * F
            print(f"Detected CSI shape: T={T}, A={A}, F={F}, A*F={in_dim}")

        # 跟訓練時一樣，把 (A*F) flatten，在每個時間步當作一個樣本
        # x_flat: [B*T, A*F]
        x_flat = csi.reshape(B * T, A * F)
        X_list.append(x_flat)
        total += x_flat.shape[0]

        if total >= args.max_samples:
            break

    if len(X_list) == 0:
        raise RuntimeError("沒有從資料集中讀到任何 CSI，請確認 root_path / Env 資料夾路徑是否正確")

    X = torch.cat(X_list, dim=0)[:args.max_samples]   # [N, in_dim]
    print(f"Collected X shape: {X.shape}")             # N ~ max_samples, dim = A*F

    X_np = X.cpu().numpy()

    # 3. Truncated SVD: in_dim -> out_dim
    print("Fitting TruncatedSVD...")
    svd = TruncatedSVD(n_components=args.out_dim)
    svd.fit(X_np)

    # svd.components_: [out_dim, in_dim]
    W = svd.components_    # numpy array
    W_torch = torch.from_numpy(W).float()  # [out_dim, in_dim]

    # 4. 存成 .pt 檔，給 model 的 Linear 使用
    torch.save(W_torch, args.save_path)
    print(f"Saved SVD projection weight to {args.save_path}")
    print(f"W_torch shape: {W_torch.shape}  (out_dim={args.out_dim}, in_dim={in_dim})")


if __name__ == '__main__':
    main()

