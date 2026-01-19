import numpy as np
import os
import re
import nibabel as nib
import tensorflow as tf
from skimage.metrics import structural_similarity
from tqdm import tqdm

# ===================== 核心配置（需根据实际路径修改） =====================
# MODEL_WEIGHTS_PATH = "/home/lengjingcheng/codes/iguane_harmonization/harmonization/iguane_weights.h5"  # 现有模型权重路径
MODEL_WEIGHTS_PATH = "/home/lengjingcheng/codes/iguane_harmonization/harmonization/my_train_sald-ixi2-abide2/latest_genUniv.h5"  # 现有模型权重路径
VAL_DATA_PATH = "/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/preprocessed/"  # 验证集路径
TARGET_SHAPE = (160, 192, 160)  # 模型输入标准尺寸

# 0.8950 my_train_sald-ixi-abide-100epoch/latest_genUniv.h5
# 0.8998 my_train_sald-ixi-abide-100epoch/best_genUniv.h5
# 0.9296 iguane_weights.h5   论文权重

# 0.8759 my_train_sald-ixi-check/latest_genUniv.h5 （60 epoch）

# 0.9483 my_train/best_genUniv.h5   ?????为啥这么高，不会是因为弄成全黑的了所以高吧？
# 0.8754 my_train2/best_genUniv.h5
# 0.9331 20个epoch 的 my_train_sald-ixi2-abide2/latest_genUniv.h5  为什么这么高，和全黑同理？看一下真实图片吧


# 0.9335 my_train_sald-ixi2-abide2  好高，是因为变成四个list了吗

# 如何验证是不是有SALD风格呢？如何验证是否有生物保真性呢？？
# ===================== 辅助函数：加载验证数据 =====================
def load_core_validation_data(val_data_path):
    """加载预处理后的验证集图像及受试者ID"""
    val_imgs = []
    val_sub_ids = []
    for file_name in os.listdir(val_data_path):
        if file_name.endswith(".nii.gz") and "sub-" in file_name:
            # 加载图像并强制标准化尺寸
            img = nib.load(os.path.join(val_data_path, file_name)).get_fdata()
            # print("已经加载图像数据 ",os.path.join(val_data_path, file_name))
            # 裁剪/填充到标准尺寸（避免尺寸不匹配）
            h, w, d = img.shape
            start_h = max(0, (h - TARGET_SHAPE[0]) // 2)
            start_w = max(0, (w - TARGET_SHAPE[1]) // 2)
            start_d = max(0, (d - TARGET_SHAPE[2]) // 2)
            img = img[start_h:start_h+TARGET_SHAPE[0],
                      start_w:start_w+TARGET_SHAPE[1],
                      start_d:start_d+TARGET_SHAPE[2]]
            # 填充不足部分为0
            pad_h = TARGET_SHAPE[0] - img.shape[0]
            pad_w = TARGET_SHAPE[1] - img.shape[1]
            pad_d = TARGET_SHAPE[2] - img.shape[2]
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            
            val_imgs.append(img)
            # 解析受试者ID
            sub_id = re.findall(r"sub-\d+", file_name)[0]
            val_sub_ids.append(sub_id)
    return np.array(val_imgs), val_sub_ids

# ===================== 核心验证函数 =====================
def eval_model(gen_univ, val_data_path):
    """加载验证集→模型推理→计算SSIM均值"""
    # 1. 加载验证数据
    val_imgs, val_sub_ids = load_core_validation_data(val_data_path)
    print(val_sub_ids)
    print(f"加载了{len(val_imgs)}张图像")
    if len(val_imgs) == 0:
        print("⚠️ 未加载到任何验证图像")
        return 0.0
    
    # 2. 生成协调后图像（冻结生成器）
    gen_univ.trainable = False
    harmonized_imgs = []
    for img in tqdm(val_imgs):
        # 预处理（与训练逻辑一致）
        mask = img > 0
        img_norm = img / 500 - 1  # 强度归一化
        img_norm[~mask] = 0
        
        # 模型推理
        img_input = np.expand_dims(img_norm, axis=(0, 4))  # [1,160,192,160,1]
        img_tensor = tf.convert_to_tensor(img_input, dtype='float32')
        harmonized = gen_univ(img_tensor, training=False).numpy().squeeze()
        
        # 后处理
        harmonized = (harmonized + 1) * 500
        harmonized[~mask] = 0
        harmonized = np.maximum(harmonized, 0)
        harmonized_imgs.append(harmonized)
    gen_univ.trainable = True

    # 3. 计算同一受试者跨站点SSIM均值
    ssim_scores = []
    unique_subs = np.unique(val_sub_ids)
    print(len(unique_subs),unique_subs)
    for sub_id in tqdm(unique_subs):
        sub_harmonized = [h for h, sid in zip(harmonized_imgs, val_sub_ids) if sid == sub_id]
        if len(sub_harmonized) < 2:
            continue
        # 遍历该受试者所有图像对
        for i in range(len(sub_harmonized)):
            for j in range(i+1, len(sub_harmonized)):
                max_val = np.percentile(np.concatenate([sub_harmonized[i].ravel(), sub_harmonized[j].ravel()]), 99)
                ssim_val = structural_similarity(
                    sub_harmonized[i], sub_harmonized[j],
                    data_range=max_val, win_size=7, multichannel=False
                )
                ssim_scores.append(ssim_val)
    
    # 4. 返回SSIM均值
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    print(f"\n✅ 验证完成：有效图像对{len(ssim_scores)}组 | 平均SSIM={avg_ssim:.4f}")
    return avg_ssim

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # 1. 初始化模型并加载权重
    print(f"🔍 加载模型权重：{MODEL_WEIGHTS_PATH}")
    import sys
    sys.path.append('..')
    from model_architectures import Generator
    gen_univ = Generator()  # 初始化论文定义的Generator架构
    gen_univ.load_weights(MODEL_WEIGHTS_PATH)
    print("✅ 模型权重加载完成")
    
    # 2. 执行验证
    print(f"\n🔍 开始验证，验证集路径：{VAL_DATA_PATH}")
    avg_ssim = eval_model(gen_univ, VAL_DATA_PATH)
    
    # 3. 输出结果
    print(f"\n📊 最终验证结果：跨站点SSIM均值 = {avg_ssim:.4f}")
