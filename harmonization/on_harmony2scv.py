import os
import pandas as pd

# 配置根目录
root_dir = "/home/lengjingcheng/datasets/ON-Harmony/ds004712-download"  # 数据集根路径
in_paths = []
out_paths = []

# 遍历BIDS结构，收集所有T1w文件
for sub_dir in os.listdir(root_dir):
    if not sub_dir.startswith("sub-"):
        continue  # 跳过非受试者目录
    sub_path = os.path.join(root_dir, sub_dir)
    for ses_dir in os.listdir(sub_path):
        if not ses_dir.startswith("ses-"):
            continue  # 跳过非会话目录
        anat_path = os.path.join(sub_path, ses_dir, "anat")
        if not os.path.exists(anat_path):
            continue  # 跳过无解剖图像的目录
        
        # 筛选T1w图像（排除defacemask掩码文件）
        for file in os.listdir(anat_path):
            if file.endswith("_T1w.nii.gz") and "defacemask" not in file:
                in_file = os.path.abspath(os.path.join(anat_path, file))
                # 输出路径：在原文件名后加_brain（标记脑提取结果）
                out_file = in_file.replace("_T1w.nii.gz", "_T1w_iguane.nii.gz")
                in_paths.append(in_file)
                out_paths.append(out_file)

# 生成CSV文件
df = pd.DataFrame({"in_paths": in_paths, "out_paths": out_paths})
df.to_csv("on-harmony4all_in_one.csv", index=False)
