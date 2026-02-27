def extract_rmse(line):
    """從一行文字提取 RMSE 數值"""
    parts = line.split("RMSE:")
    if len(parts) < 2:
        return None
    return float(parts[1].split(",")[0].strip())

def compare_rmse(file1, file2, output_file="rmse_diff.txt"):
    # 讀取檔案
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # 只取前 4937 行
    max_lines = min(len(lines1), len(lines2), 4824)
    lines1 = lines1[:max_lines]
    lines2 = lines2[:max_lines]

    results = []
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        rmse1 = extract_rmse(l1)
        rmse2 = extract_rmse(l2)
        if rmse1 is not None and rmse2 is not None:
            diff = rmse1 - rmse2
            results.append((i+1, rmse1, rmse2, diff, l1.strip(), l2.strip()))

    # 排序 (依照差值從大到小)
    results.sort(key=lambda x: x[3], reverse=True)

    # 輸出到新檔案
    with open(output_file, "w") as f:
        for idx, r1, r2, diff, l1, l2 in results:
            f.write(
                f"Line {idx}: RMSE1={r1}, RMSE2={r2}, Diff={diff}\n"
                f"  File1: {l1}\n"
                f"  File2: {l2}\n\n"
            )

    print(f"✅ 已完成，取前 {max_lines} 行，結果已存到 {output_file}")

# === 使用方式 ===
compare_rmse(
    "/media/main/HDD/yo/GraphMLP-main/output_ViTPose_test9/rmse_results_adj_vitpose.txt",
    "/media/main/HDD/yo/GraphMLP-main/output_ViTPose_test10/rmse_results_adj_vitpose.txt"
)

