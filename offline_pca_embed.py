# offline_pca_embed.py
import os
import glob
import argparse
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
import pywt
from sklearn.decomposition import IncrementalPCA


# ===========================
# 1. 跟原訓練程式一樣的前處理
# ===========================
def butter_filter(data, cutoff=10, fs=1000, order=4):
    """ 對每個天線資料進行 Butterworth 低通濾波（逐 channel）"""
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):      # 51 time
        for j in range(data.shape[1]):  # 6 antennas
            filtered[i, j] = filtfilt(b, a, data[i, j])
    return filtered


def dwt_denoise(data, wavelet='db4', level=1):
    denoised = np.zeros_like(data)
    # data: [T, A, F]
    for j in range(data.shape[1]):      # antennas
        for k in range(data.shape[2]):  # subcarriers
            signal = data[:, j, k]      # (T,)
            coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)

            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

            coeffs[1:] = [
                pywt.threshold(c, value=uthresh, mode='soft')
                if np.max(np.abs(c)) > 1e-12 else np.zeros_like(c)
                for c in coeffs[1:]
            ]

            recon = pywt.waverec(coeffs, wavelet, mode='periodization')
            denoised[:, j, k] = recon[:signal.shape[0]]
    return denoised


def normalize(data):
    """ 對每個 channel zero-mean, unit-std （沿時間維做）"""
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True) + 1e-8
    return (data - mean) / std


# ===========================
# 2. CSI_KPT_Dataset：沿用你的版本
# ===========================
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

        csi_npz = np.load(csi_path)
        mag = csi_npz["mag"].astype(np.float32)
        pha = csi_npz["pha"].astype(np.float32)

        # ====== 跟你原本一樣：目前只 normalize ======
        # mag = normalize(butter_filter(mag))
        # pha = normalize(butter_filter(pha))
        # mag = normalize(dwt_denoise(mag))
        # pha = normalize(dwt_denoise(pha))

        mag = normalize(mag)
        pha = normalize(pha)

        csi = np.concatenate([mag, pha], axis=2)   # [T, A, 4050]

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
# 3. 建 DataLoader + 抽樣資料做 PCA
# ===========================
def build_csi_loader(root_path, batch_size=8, workers=4):
    env_roots = [os.path.join(root_path, "Env1_train")]
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
                        help="PCA 降維後的維度 (要跟 model.args.channel 一致)")
    parser.add_argument('--max_samples', type=int, default=50000,
                        help="最多抽多少個 time step 來做 PCA (每個 time step 算一筆)")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='pca_24300_to_512_50000_test1.npz',
                        help="輸出檔名，會包含 mean 與 components")
    parser.add_argument('--ipca_batch', type=int, default=2048,
                        help="IncrementalPCA 的 batch_size（對 partial_fit）")
    args = parser.parse_args()

    #torch.manual_seed(1)
    #np.random.seed(1)

    loader, dataset = build_csi_loader(args.root_path, args.batch_size, args.workers)
    print(f"Dataset size: {len(dataset)} samples")

    ipca = IncrementalPCA(n_components=args.out_dim, batch_size=args.ipca_batch)

    total = 0
    in_dim = None

    print("Fitting IncrementalPCA (partial_fit) ...")
    for csi, _ in loader:  # csi: [B, T, A, F]
        csi = csi.float()
        B, T, A, F = csi.shape

        if in_dim is None:
            in_dim = A * F
            print(f"Detected CSI shape: T={T}, A={A}, F={F}, A*F={in_dim}")

        # [B*T, 24300]，每個 time step 當一筆樣本
        x_flat = csi.reshape(B * T, A * F).cpu().numpy().astype(np.float32)

        # 只取到 max_samples 為止
        remain = args.max_samples - total
        if remain <= 0:
            break
        if x_flat.shape[0] > remain:
            x_flat = x_flat[:remain]

        # ✅ PCA 的核心：partial_fit 會自己累積 mean/cov
        ipca.partial_fit(x_flat)

        total += x_flat.shape[0]
        if total >= args.max_samples:
            break

    if total == 0:
        raise RuntimeError("沒有從資料集中讀到任何 CSI，請確認 root_path / Env 資料夾路徑是否正確")

    print(f"Total samples used for PCA: {total}")

    # PCA 結果：
    # components_: [out_dim, in_dim]
    # mean_:       [in_dim]
    components = ipca.components_.astype(np.float32)
    mean = ipca.mean_.astype(np.float32)

    np.savez(
        args.save_path,
        mean=mean,
        components=components,
        explained_variance=ipca.explained_variance_.astype(np.float32),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
    )

    print(f"Saved PCA to {args.save_path}")
    print(f"mean shape: {mean.shape}  (in_dim={in_dim})")
    print(f"components shape: {components.shape}  (out_dim={args.out_dim}, in_dim={in_dim})")


if __name__ == '__main__':
    main()

