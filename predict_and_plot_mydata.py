#GT沒有點座標就不計算
import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import glob
from PIL import Image
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from model.graphmlp_origin import Model
#from model.graphmlp import Model
#from model.graphmlp_attention import Model
from common.arguments import parse_args
import common.eval_cal as eval_cal

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 5), (12, 6),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def pad_A_to_6(x):
    # x: [T, A, F]
    T, A, F = x.shape
    if A == 6:
        return x
    if A != 4:
        raise ValueError(f"Expected A=4 or 6, but got A={A}, shape={x.shape}")

    # 用平均天線補兩根（比補 0 穩）
    mean_ant = x.mean(axis=1, keepdims=True)      # [T,1,F]
    pad = np.repeat(mean_ant, 2, axis=1)          # [T,2,F]
    return np.concatenate([x, pad], axis=1)       # [T,6,F]
def dwt_denoise(data, wavelet='db4', level=1):
    denoised = np.zeros_like(data)
    for j in range(data.shape[1]):          # 6 antennas
        for k in range(data.shape[2]):      # T subcarriers
            signal = data[:, j, k]          # 取出 (51,) → 一條時間序列
            coeffs = pywt.wavedec(
                signal, wavelet, mode='periodization', level=level
            )

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
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True) + 1e-8
    return (data - mean) / std

def Rmse(predicted, target):
    if isinstance(predicted, np.ndarray):
        predicted = torch.from_numpy(predicted)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    return torch.mean(torch.norm(predicted - target, dim=-1))

def masked_rmse(predicted, target):
    mask = ~(np.all(target == 0, axis=-1))
    if mask.sum() == 0:
        return 0.0
    sq_err = np.sum((predicted[mask] - target[mask]) ** 2, axis=-1)
    return np.sqrt(np.mean(sq_err))

def avg_pck_result_multi(predicted, target, alphas=[0.1,0.2,0.3,0.4,0.5], right_shoulder=6, left_hip=11):
    """
    多 alpha PCK
    回傳 dict {alpha: 值}
    """
    assert predicted.shape == target.shape
    B, J, _ = predicted.shape

    ls_valid = (target[:, right_shoulder].abs().sum(dim=-1) > 0)
    rs_valid = (target[:, left_hip].abs().sum(dim=-1) > 0)
    sample_valid = ls_valid & rs_valid

    shoulder_dists = torch.norm(
        target[:, right_shoulder] - target[:, left_hip], dim=-1
    ).clamp(min=1e-6)

    dists = torch.norm(predicted - target, dim=-1)
    normalized = dists / shoulder_dists[:, None]

    joint_valid = (target.abs().sum(dim=-1) > 0) & sample_valid[:, None]
    if joint_valid.sum() == 0:
        return {a: 0.0 for a in alphas}

    results = {}
    for a in alphas:
        correct = (normalized <= a) & joint_valid
        results[a] = (correct.float().sum() / joint_valid.float().sum()).item()
    return results

def collect_npz_files(root="/media/main/HDD/yo/dataset/Env1_test/npy_dwt", pattern="*.npz"):
    """
    遞迴收集 root 底下所有 npz 檔案
    """
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    
def build_pair_paths(csi_path, csi_root, img_root, gt_root):
    """
    把 csi_path 相對於 csi_root 的相對路徑取出，然後映射到 img_root / gt_root
    例如：
      csi_root/.../a/b/xxx.npz
      -> img_root/.../a/b/xxx.jpg
      -> gt_root/.../a/b/xxx.npz
    """
    rel = os.path.relpath(csi_path, csi_root)                 # a/b/xxx.npz
    rel_no_ext = os.path.splitext(rel)[0]                     # a/b/xxx

    image_path = os.path.join(img_root, rel_no_ext + ".jpg")
    gt_path    = os.path.join(gt_root, rel_no_ext + ".npz")
    return image_path, gt_path


def main():
    args = parse_args()
    args.train = 0
    args.dataset = 'csi'
    args.n_joints = 17
    args.out_channels = 2
    args.frames = 1
    args.pad = 0

    model = Model(args).cuda()
    checkpoint = torch.load('/media/main/HDD/yo/GraphMLP-main/checkpoint/1229_0003_30_1/model_37_6654.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    output_dir = "/media/main/HDD/yo/GraphMLP-main/output_ViTPose_mydata"
    os.makedirs(output_dir, exist_ok=True)
    
    csi_root = "/media/main/HDD/yo/dataset/stand5_test/npy_merged"
    img_root = "/media/main/HDD/yo/dataset/stand5_test/rgb"
    gt_root  = "/media/main/HDD/yo/dataset/stand5_test/kpt_npz_vitpose"

    sampled_files = collect_npz_files(csi_root)
    results = []

    J = 17
    joint_names = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]
    RS, LH = 6, 11
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]

    # per-joint 統計
    per_joint_sqerr_sum = np.zeros(J, dtype=np.float64)
    per_joint_pck_correct = {a: np.zeros(J, dtype=np.float64) for a in alphas}
    per_joint_count = 0

    for idx, csi_path in enumerate(tqdm(sampled_files, desc="🔍 推論中")):
        filename = os.path.splitext(os.path.basename(csi_path))[0]
        save_prefix = os.path.join(output_dir, filename)
        
        image_path, gt_path = build_pair_paths(csi_path, csi_root, img_root, gt_root)

        csi_npz = np.load(csi_path)
        mag = normalize(dwt_denoise(csi_npz["mag"].astype(np.float32)))
        pha = normalize(csi_npz["phase"].astype(np.float32))
        x = np.concatenate([mag, pha], axis=2)   # [T, A, 4050]  (假設原本 2025)
        x = pad_A_to_6(x)                        # [T, 6, 4050]
        csi_input = torch.from_numpy(x).unsqueeze(0).float().cuda()  # [1,T,6,4050]

        with torch.no_grad():
            pred = model(csi_input).squeeze(0).cpu().numpy()

        if os.path.exists(gt_path):
            gt_npz = np.load(gt_path)
            if "keypoints" in gt_npz:
                gt = gt_npz["keypoints"]
                gt = gt[0] if gt.ndim == 3 else gt

                rmse = masked_rmse(pred, gt)

                pred_tensor = torch.from_numpy(pred).unsqueeze(0).float()
                gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()
                pck_dict = avg_pck_result_multi(pred_tensor, gt_tensor, alphas, right_shoulder=RS, left_hip=LH)

                results.append((filename, float(rmse), pck_dict))

                mask = ~(np.all(gt == 0, axis=-1))

                sqerr_per_joint = ((pred - gt) ** 2).sum(axis=-1)
                per_joint_sqerr_sum += sqerr_per_joint * mask

                ref_dist = np.linalg.norm(gt[RS] - gt[LH]) + 1e-9
                dist_per_joint = np.sqrt(sqerr_per_joint)
                normalized = dist_per_joint / ref_dist
                for a in alphas:
                    hit = (normalized <= a).astype(np.float64)
                    per_joint_pck_correct[a] += hit * mask

                per_joint_count += 1
            else:
                results.append((filename, None, None))
        else:
            results.append((filename, None, None))

        if os.path.exists(image_path):
            image = Image.open(image_path)
            plt.imshow(image)
            for i, (x, y) in enumerate(pred):
                plt.plot(x, y, 'ro', markersize=2)
                plt.text(x + 5, y - 5, f"{i}", fontsize=4, color='yellow')
            for i, j in COCO_EDGES:
                if i < len(pred) and j < len(pred):
                    x1, y1 = pred[i]
                    x2, y2 = pred[j]
                    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1)
            plt.axis('off')
            plt.savefig(save_prefix + "_adj_vitpose.png", bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

    rmse_path = os.path.join(output_dir, "rmse_results_adj_vitpose.txt")
    with open(rmse_path, "w") as f:
        for fname, rmse, pck_dict in results:
            if rmse is not None:
                pck_str = ", ".join([f"PCK@{a}: {pck_dict[a]:.3f}" for a in alphas])
                f.write(f"{fname}, RMSE: {rmse:.2f}, {pck_str}\n")
            else:
                f.write(f"{fname}, RMSE: N/A, PCK: N/A (GT not found)\n")

    valid_rmses = [rmse for _, rmse, _ in results if rmse is not None]
    if valid_rmses:
        avg_rmse = sum(valid_rmses) / len(valid_rmses)
        avg_pcks = {a: np.mean([p[a] for _, _, p in results if p is not None]) for a in alphas}
        with open(rmse_path, "a") as f:
            f.write("\nAverage RMSE: {:.2f}\n".format(avg_rmse))
            for a in alphas:
                f.write("Average PCK@{:.1f}: {:.3f}\n".format(a, avg_pcks[a]))

    # per-joint 統計輸出
    if per_joint_count > 0:
        per_joint_rmse = np.sqrt(per_joint_sqerr_sum / per_joint_count)
        per_joint_pck = {a: per_joint_pck_correct[a] / per_joint_count for a in alphas}

        per_joint_path = os.path.join(output_dir, "per_joint_metrics_vitpose.txt")
        with open(per_joint_path, "w") as jf:
            header = "Joint, RMSE(px), " + ", ".join([f"PCK@{a}" for a in alphas]) + "\n"
            jf.write(header)
            for j in range(J):
                pck_vals = ", ".join([f"{per_joint_pck[a][j]:.3f}" for a in alphas])
                jf.write(f"{j:02d}-{joint_names[j]}, {per_joint_rmse[j]:.2f}, {pck_vals}\n")

if __name__ == "__main__":
    main()

