import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CSIDataset(Dataset):
    def __init__(self, root_path):
        self.files = []
        self.kpt_files = []

        npy_paths = sorted(glob.glob(os.path.join(root_path, "Env*/npy/**/*.npz"), recursive=True))

        for npy_path in npy_paths:
            # 對應的 keypoints 檔案路徑
            kpt_path = npy_path.replace("/npy/", "/kpt_npz/")

            if os.path.exists(kpt_path):
                self.files.append(npy_path)
                self.kpt_files.append(kpt_path)

        assert len(self.files) > 0, "❌ 找不到任何有對應 kpt_npz 的樣本"
        print(f"✅ 共載入 {len(self.files)} 筆樣本")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        mag = data['mag'].astype(np.float32)   # shape (51, 6, 2025)
        pha = data['pha'].astype(np.float32)   # shape (51, 6, 2025)
        x = np.concatenate([mag, pha], axis=-1)  # shape (51, 6, 4050)
        x = x.reshape(51, -1, 4050)  # 6 antennas → 306 = 6 * 51
        x = torch.from_numpy(x)     # → Tensor: (51, 306, 4050)


        kpt_data = np.load(self.kpt_files[idx])
        y = kpt_data['keypoints'].astype(np.float32)  # shape: (1, 17, 2) or (N, 17, 2)
        y = y[0]  # 若只用第 1 個人的 keypoints
        y = torch.from_numpy(y)  # shape: (17, 2)

        return x, y
