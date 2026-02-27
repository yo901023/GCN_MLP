import numpy as np
import pandas as pd

data = np.load(r'E:\GraphMLP-main\demo\output\sample_video\output_3D\output_keypoints_3d.npz')
pose_3d = data['reconstruction']  # shape: (frames, 17, 3)

rows = []
for frame_idx, frame in enumerate(pose_3d):
    for joint_idx, joint in enumerate(frame):
        x, y, z = joint
        rows.append([frame_idx, joint_idx, x, y, z])

df = pd.DataFrame(rows, columns=['frame', 'joint', 'x', 'y', 'z'])
df.to_csv('pose3d_output.csv', index=False)
print("✅ 已儲存為 pose3d_output.csv")
