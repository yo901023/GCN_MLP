import argparse
import numpy as np

def load_npz(path):
    z = np.load(path)
    need = ["mean", "components", "explained_variance_ratio"]
    for k in need:
        if k not in z.files:
            raise ValueError(f"{path} missing key: {k}. found={z.files}")
    return {
        "mean": z["mean"].astype(np.float64),
        "C": z["components"].astype(np.float64),  # (K, D)
        "evr": z["explained_variance_ratio"].astype(np.float64),
        "path": path
    }

def report_evr(a, b):
    cum_a = np.cumsum(a["evr"])
    cum_b = np.cumsum(b["evr"])
    print("\n[1] Cumulative explained variance (EVR)")
    for k in [10, 50, 100, 256, len(a["evr"])]:
        da = cum_a[k-1]; db = cum_b[k-1]
        print(f"  @ {k:>4d}: {da:.6f} vs {db:.6f} | diff={abs(da-db):.6f}")
    print(f"  total@{len(a['evr'])}: {cum_a[-1]:.6f} vs {cum_b[-1]:.6f} | diff={abs(cum_a[-1]-cum_b[-1]):.6f}")

def report_subspace(a, b):
    C1, C2 = a["C"], b["C"]  # (K,D)
    if C1.shape != C2.shape:
        raise ValueError(f"components shape mismatch: {C1.shape} vs {C2.shape}")

    # Subspace similarity via singular values of C1*C2^T
    M = C1 @ C2.T   # (K,K)
    s = np.linalg.svd(M, compute_uv=False)  # in [0,1] ideally
    print("\n[2] Subspace similarity (principal angles cosines)")
    print(f"  mean={s.mean():.6f}, median={np.median(s):.6f}, min={s.min():.6f}, max={s.max():.6f}")

    # Also show “best match per component” (sign-invariant), helpful intuition
    A = np.abs(M)
    best = A.max(axis=1)  # each comp in run1 best aligned with some comp in run2
    print("\n[2b] Per-component best alignment |C1_i · C2_j|")
    print(f"  mean={best.mean():.6f}, median={np.median(best):.6f}, min={best.min():.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    args = ap.parse_args()

    a = load_npz(args.p1)
    b = load_npz(args.p2)

    print("PCA-1:", a["path"], "components", a["C"].shape, "mean", a["mean"].shape)
    print("PCA-2:", b["path"], "components", b["C"].shape, "mean", b["mean"].shape)

    report_evr(a, b)
    report_subspace(a, b)

    print("\n[Interpretation tips]")
    print("  - EVR diff at 512 < 0.01 代表兩次學到的『變異趨勢』非常一致。")
    print("  - Subspace similarity mean > 0.9 且 min 不要太低，代表 512 維子空間很接近。")

if __name__ == "__main__":
    main()

