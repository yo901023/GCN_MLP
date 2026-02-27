import os
import io
import numpy as np
import zipfile
from torch.utils.data import Dataset
import torch

class MultiCSIZipDataset(Dataset):
    def __init__(self, npz_paths):
        self.samples = []

        for npz_path in npz_paths:
            zf = zipfile.ZipFile(npz_path, 'r')
            # ✅ 使用 io.BytesIO + np.load 來正確讀取 keys.npy
            keys_buf = io.BytesIO(zf.read('keys.npy'))
            keys = np.load(keys_buf, allow_pickle=True)
            for idx in range(len(keys)):
                self.samples.append((zf, f'csi_{idx}.npy', f'kpt_{idx}.npy'))

        print(f"✅ 總共載入 {len(self.samples)} 筆樣本，來自 {len(npz_paths)} 個 npz 檔")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx): 
        zf, csi_name, kpt_name = self.samples[idx]

        try:
            with zf.open(csi_name) as csi_file:
                csi_data = np.load(io.BytesIO(csi_file.read()))
                csi_tensor = torch.tensor(csi_data, dtype=torch.float32)

            with zf.open(kpt_name) as kpt_file:
                kpt_data = np.load(io.BytesIO(kpt_file.read()))
                kpt_tensor = torch.tensor(kpt_data, dtype=torch.float32)

            # 可選的 shape 檢查與跳過
            if csi_tensor.shape != (51, 6, 4050) or kpt_tensor.shape != (1, 17, 2):
                raise ValueError(f"⛔ Shape mismatch: {csi_tensor.shape}, {kpt_tensor.shape}")

            return csi_tensor, kpt_tensor

        except Exception as e:
            print(f"⚠️ 資料損壞，跳過：{zf.filename} / {csi_name}，錯誤：{e}")
            return self.__getitem__((idx + 1) % len(self.samples))  # 遞迴取下一筆
