import os
import re
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.stats as stats
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import tensorflow as tf
from model_architectures import Generator
from tqdm import tqdm  # 进度条，提升可视化体验

# ===================== 1. 配置全局参数（适配BIDS结构） =====================
class Config:
    # 模型权重路径
    WEIGHTS_PATH = './iguane_weights.h5'
    # 旅行受试者数据集根目录（核心适配点）
    TRAVEL_DATA_ROOT = '/home/lengjingcheng/datasets/ON-Harmony/ds004712-download'  # 替换为你的数据集根目录
    # 输出目录（协调后图像+验证结果）
    OUTPUT_DIR = './travel_subjects_harmonized'
    # 预处理参数（与论文一致）
    MEDIAN_NORM_VALUE = 500
    # SSIM计算参数
    SSIM_DATA_RANGE_PERCENTILE = 99  # 论文中99百分位数作为强度范围
    # 统计检验参数
    ALPHA = 0.05  # 显著性水平
    # BIDS格式过滤参数
    T1W_SUFFIX = '_T1w.nii.gz'  # 目标图像后缀
    EXCLUDE_SUFFIX = '_defacemask.nii.gz'  # 排除的掩码文件

# 创建输出目录
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ===================== 2. 工具函数：解析BIDS结构生成元数据 =====================
def parse_bids_travel_dataset(data_root):
    """
    自动遍历BIDS格式的旅行受试者数据集，生成元数据CSV
    :param data_root: 数据集根目录
    :return: 元数据DataFrame、保存的CSV路径
    """
    metadata = []
    # 遍历所有受试者目录（sub-xxxx）
    for sub_dir in tqdm(os.listdir(data_root), desc="解析BIDS目录"):
        if not sub_dir.startswith('sub-'):
            continue  # 过滤非受试者目录
        sub_id = sub_dir  # 受试者ID（如sub-16981）
        
        # 遍历该受试者下的所有扫描场次（ses-xxxx，对应不同扫描仪）
        sub_full_path = os.path.join(data_root, sub_dir)
        for ses_dir in os.listdir(sub_full_path):
            if not ses_dir.startswith('ses-'):
                continue  # 过滤非场次目录
            scanner_id = ses_dir  # 扫描场次=扫描仪ID（旅行受试者核心逻辑）
            
            # 遍历anat目录
            anat_path = os.path.join(sub_full_path, ses_dir, 'anat')
            if not os.path.exists(anat_path):
                continue  # 跳过无anat目录的场次
            
            # 查找T1w图像文件
            for file_name in os.listdir(anat_path):
                # 过滤目标文件：仅保留T1w.nii.gz，排除defacemask
                if file_name.endswith(Config.T1W_SUFFIX) and not file_name.endswith(Config.EXCLUDE_SUFFIX):
                    mri_path = os.path.join(anat_path, file_name)
                    # 记录元数据
                    metadata.append({
                        'sub_id': sub_id,
                        'scanner_id': scanner_id,
                        'mri_path': mri_path
                    })
    
    # 生成DataFrame并保存
    df = pd.DataFrame(metadata)
    # 去重（避免重复文件）
    df = df.drop_duplicates(subset=['sub_id', 'scanner_id', 'mri_path']).reset_index(drop=True)
    # 保存元数据CSV
    metadata_csv = os.path.join(Config.OUTPUT_DIR, 'travel_subjects_bids_metadata.csv')
    df.to_csv(metadata_csv, index=False)
    print(f"✅ BIDS元数据已生成：{metadata_csv}")
    return df, metadata_csv

# ===================== 3. 初始化模型（支持GPU混合精度） =====================
def init_model(weights_path):
    """初始化IGUANe生成器模型"""
    # GPU配置
    if len(tf.config.list_physical_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("✅ GPU可用，启用混合精度推理")
    else:
        print("ℹ️ 仅使用CPU推理")
    
    # 加载模型
    gen = Generator()
    gen.load_weights(weights_path)
    return gen

# ===================== 4. 单张图像协调推理（复用原有逻辑，适配BIDS路径） =====================
def harmonize_mri(mri_path, dest_path, gen):
    """
    对单张MRI图像执行IGUANe协调（适配BIDS格式图像）
    :param mri_path: 输入MRI路径（NIfTI格式）
    :param dest_path: 协调后保存路径
    :param gen: 加载好的Generator模型
    :return: 协调后的图像数据（numpy数组）
    """
    try:
        # 加载图像
        mri = nib.load(mri_path)
        data = mri.get_fdata()
        affine = mri.affine
        header = mri.header
        
        # 预处理（与论文一致）
        mask = data > 0  # 脑区掩码（背景为0）
        data = data / Config.MEDIAN_NORM_VALUE - 1  # 强度归一化
        data[~mask] = 0  # 背景置0
        
        # 模型输入格式：(batch, H, W, D, channel)
        data_input = np.expand_dims(data, axis=(0, 4))
        tensor_input = tf.convert_to_tensor(data_input, dtype='float32')
        
        # 模型推理（training=False关闭Dropout等训练层）
        harmonized_data = gen(tensor_input, training=False).numpy().squeeze()
        
        # 后处理：恢复强度范围
        harmonized_data = (harmonized_data + 1) * Config.MEDIAN_NORM_VALUE
        harmonized_data[~mask] = 0  # 背景置0
        harmonized_data = np.maximum(harmonized_data, 0)  # 确保非负
        
        # 保存协调后图像（保持BIDS风格命名）
        harmonized_mri = nib.Nifti1Image(harmonized_data, affine=affine, header=header)
        nib.save(harmonized_mri, dest_path)
        
        return harmonized_data
    except Exception as e:
        print(f"❌ 处理图像失败 {mri_path}：{str(e)}")
        return None

# ===================== 5. 计算SSIM（严格对齐论文逻辑） =====================
def calculate_ssim(img1, img2):
    """
    计算两张MRI图像的SSIM（适配论文中的参数）
    :param img1: 图像1数据（3D numpy数组）
    :param img2: 图像2数据（3D numpy数组）
    :return: SSIM值
    """
    # 确保图像非负（论文中SSIM仅用于非负图像）
    img1 = np.maximum(img1, 0)
    img2 = np.maximum(img2, 0)
    
    # 计算99百分位数作为数据范围（减少极端值影响）
    max_val = np.percentile(np.concatenate([img1.ravel(), img2.ravel()]), Config.SSIM_DATA_RANGE_PERCENTILE)
    data_range = max_val - 0  # 最小值为0
    
    # 计算3D SSIM（适配脑图像的空间结构）
    ssim_value = ssim(
        img1, img2,
        data_range=data_range,
        win_size=7,  # 论文常用窗口大小（奇数）
        multichannel=False,
        gaussian_weights=True,  # 高斯权重更贴合脑结构
        sigma=1.5
    )
    return ssim_value

# ===================== 6. 计算同受试者跨扫描仪的SSIM（核心验证步骤） =====================
def compute_subject_ssim(df, sub_ids, harmonized=False):
    """
    计算每个受试者在不同扫描仪下的图像对SSIM
    :param df: 元数据DataFrame
    :param sub_ids: 受试者ID列表
    :param harmonized: 是否使用协调后图像
    :return: ssim_results: {sub_id: [ssim1, ssim2, ...]}
    """
    ssim_results = {}
    
    for sub_id in tqdm(sub_ids, desc=f"计算SSIM（协调后：{harmonized}）"):
        # 获取该受试者的所有图像路径
        sub_df = df[df['sub_id'] == sub_id].reset_index(drop=True)
        mri_paths = sub_df['harmonized_path'].tolist() if harmonized else sub_df['mri_path'].tolist()
        n_imgs = len(mri_paths)
        
        if n_imgs < 2:
            print(f"⚠️ 受试者{sub_id}仅{str(n_imgs)}张图像，跳过SSIM计算")
            ssim_results[sub_id] = []
            continue
        
        # 加载该受试者的所有图像数据
        sub_imgs = []
        valid_paths = []
        for path in mri_paths:
            try:
                img = nib.load(path).get_fdata()
                sub_imgs.append(img)
                valid_paths.append(path)
            except:
                print(f"⚠️ 加载图像失败 {path}，跳过")
                continue
        
        # 计算所有图像对的SSIM（i<j避免重复）
        sub_ssim = []
        n_valid = len(sub_imgs)
        for i in range(n_valid):
            for j in range(i+1, n_valid):
                ssim_val = calculate_ssim(sub_imgs[i], sub_imgs[j])
                sub_ssim.append(ssim_val)
        
        ssim_results[sub_id] = sub_ssim
    
    return ssim_results

# ===================== 7. 计算欧氏距离相关性（验证个体差异保留） =====================
def compute_euclidean_correlation(df, scanner_ids, harmonized=False):
    """
    计算同扫描仪内个体间欧氏距离的Pearson相关系数（处理前后）
    :param df: 元数据DataFrame
    :param scanner_ids: 扫描仪ID列表
    :param harmonized: 是否使用协调后图像
    :return: 若harmonized=True返回(均值, 标准差)；否则返回{scanner_id: 距离数组}
    """
    # 非协调模式：返回各扫描仪的原始距离
    if not harmonized:
        corr_results = {}
        for scanner_id in tqdm(scanner_ids, desc="计算原始欧氏距离"):
            # 获取该扫描仪下的所有图像
            scanner_df = df[df['scanner_id'] == scanner_id].reset_index(drop=True)
            mri_paths = scanner_df['mri_path'].tolist()
            n_imgs = len(mri_paths)
            
            if n_imgs < 2:
                corr_results[scanner_id] = np.array([])
                continue
            
            # 加载图像并提取脑区向量
            img_vectors = []
            for path in mri_paths:
                try:
                    img = nib.load(path).get_fdata()
                    mask = img > 0  # 仅保留脑区
                    img_flat = img[mask].ravel()
                    img_vectors.append(img_flat)
                except:
                    print(f"⚠️ 加载图像失败 {path}，跳过")
                    continue
            
            # 计算欧氏距离矩阵
            n_valid = len(img_vectors)
            dist_matrix = np.zeros((n_valid, n_valid))
            for i in range(n_valid):
                for j in range(n_valid):
                    if i != j:
                        dist_matrix[i, j] = np.linalg.norm(img_vectors[i] - img_vectors[j])
            
            # 提取上三角矩阵（避免重复）
            dist_flat = dist_matrix[np.triu_indices(n_valid, k=1)]
            corr_results[scanner_id] = dist_flat
        return corr_results
    
    # 协调模式：计算与原始距离的相关性
    else:
        # 先获取原始距离
        orig_dist = compute_euclidean_correlation(df, scanner_ids, harmonized=False)
        corr_list = []
        
        for scanner_id in tqdm(scanner_ids, desc="计算协调后欧氏距离相关性"):
            # 获取协调后图像路径
            scanner_df = df[df['scanner_id'] == scanner_id].reset_index(drop=True)
            mri_paths = scanner_df['harmonized_path'].tolist()
            n_imgs = len(mri_paths)
            
            if n_imgs < 2 or len(orig_dist[scanner_id]) == 0:
                continue
            
            # 加载协调后图像并提取脑区向量
            img_vectors = []
            for path in mri_paths:
                try:
                    img = nib.load(path).get_fdata()
                    mask = img > 0
                    img_flat = img[mask].ravel()
                    img_vectors.append(img_flat)
                except:
                    print(f"⚠️ 加载协调后图像失败 {path}，跳过")
                    continue
            
            # 计算协调后距离矩阵
            n_valid = len(img_vectors)
            dist_matrix = np.zeros((n_valid, n_valid))
            for i in range(n_valid):
                for j in range(n_valid):
                    if i != j:
                        dist_matrix[i, j] = np.linalg.norm(img_vectors[i] - img_vectors[j])
            dist_flat = dist_matrix[np.triu_indices(n_valid, k=1)]
            
            # 确保原始距离和协调后距离长度一致
            min_len = min(len(orig_dist[scanner_id]), len(dist_flat))
            if min_len < 2:
                continue
            
            # 计算Pearson相关系数
            corr, _ = stats.pearsonr(orig_dist[scanner_id][:min_len], dist_flat[:min_len])
            if not np.isnan(corr):  # 排除NaN值
                corr_list.append(corr)
        
        # 计算均值和标准差
        corr_mean = np.mean(corr_list) if corr_list else 0
        corr_std = np.std(corr_list) if corr_list else 0
        return corr_mean, corr_std

# ===================== 8. 统计检验（Wilcoxon + Benjamini-Hochberg） =====================
def statistical_test(ssim_original, ssim_harmonized):
    """
    执行配对Wilcoxon符号秩检验 + Benjamini-Hochberg校正
    :param ssim_original: 原始图像SSIM字典 {sub_id: [ssim列表]}
    :param ssim_harmonized: 协调后图像SSIM字典 {sub_id: [ssim列表]}
    :return: 检验结果字典
    """
    # 按受试者聚合SSIM差值
    sub_diff = []
    for sub_id in ssim_original.keys():
        orig_ssim = ssim_original[sub_id]
        harm_ssim = ssim_harmonized[sub_id]
        
        if len(orig_ssim) == 0 or len(harm_ssim) == 0:
            continue
        
        # 每个受试者的SSIM均值
        orig_mean = np.mean(orig_ssim)
        harm_mean = np.mean(harm_ssim)
        sub_diff.append(harm_mean - orig_mean)
    
    if len(sub_diff) == 0:
        print("⚠️ 无足够数据进行统计检验")
        return None
    
    # Wilcoxon符号秩检验（配对、双尾）
    stat, p_value = wilcoxon(sub_diff, alternative='two-sided')
    
    # Benjamini-Hochberg校正（多检验校正）
    _, p_corrected, _, _ = multipletests([p_value], alpha=Config.ALPHA, method='fdr_bh')
    
    # 结果整理
    return {
        'wilcoxon_stat': stat,
        'raw_p_value': p_value,
        'corrected_p_value': p_corrected[0],
        'is_significant': p_corrected[0] < Config.ALPHA,
        'mean_ssim_diff': np.mean(sub_diff),
        'std_ssim_diff': np.std(sub_diff),
        'n_subjects': len(sub_diff)
    }

# ===================== 9. 主流程执行（适配BIDS结构） =====================
def main():
    # 步骤1：解析BIDS目录，生成元数据
    df, metadata_csv = parse_bids_travel_dataset(Config.TRAVEL_DATA_ROOT)
    if len(df) == 0:
        print("❌ 未解析到任何MRI图像，请检查数据集路径")
        return
    
    # 步骤2：初始化模型
    gen = init_model(Config.WEIGHTS_PATH)
    
    # 步骤3：批量协调旅行受试者图像（生成BIDS风格的输出路径）
    print("\n===== 开始协调旅行受试者图像 =====")
    harmonized_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="协调图像"):
        mri_path = row['mri_path']
        sub_id = row['sub_id']
        scanner_id = row['scanner_id']
        
        # 生成协调后路径（保持BIDS风格）
        file_name = os.path.basename(mri_path)
        harmonized_file = file_name.replace(Config.T1W_SUFFIX, '_harmonized' + Config.T1W_SUFFIX)
        harmonized_path = os.path.join(Config.OUTPUT_DIR, sub_id, scanner_id, 'anat', harmonized_file)
        # 创建输出目录
        os.makedirs(os.path.dirname(harmonized_path), exist_ok=True)
        
        # 执行协调
        harmonize_mri(mri_path, harmonized_path, gen)
        harmonized_paths.append(harmonized_path)
    
    # 将协调后路径添加到DataFrame
    df['harmonized_path'] = harmonized_paths
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'travel_subjects_harmonized_metadata.csv'), index=False)
    
    # 步骤4：提取核心列表（受试者/扫描仪ID）
    sub_ids = sorted(df['sub_id'].unique())
    scanner_ids = sorted(df['scanner_id'].unique())
    print(f"\n✅ 数据集概览：{len(sub_ids)}个受试者，{len(scanner_ids)}个扫描仪")
    
    # 步骤5：计算原始图像SSIM
    print("\n===== 计算原始图像SSIM =====")
    ssim_original = compute_subject_ssim(df, sub_ids, harmonized=False)
    
    # 步骤6：计算协调后图像SSIM
    print("\n===== 计算协调后图像SSIM =====")
    ssim_harmonized = compute_subject_ssim(df, sub_ids, harmonized=True)
    
    # 步骤7：统计检验（SSIM差异）
    print("\n===== 执行统计检验 =====")
    stat_result = statistical_test(ssim_original, ssim_harmonized)
    
    # 步骤8：计算欧氏距离相关性
    print("\n===== 计算欧氏距离相关性 =====")
    corr_mean, corr_std = compute_euclidean_correlation(df, scanner_ids, harmonized=True)
    
    # 步骤9：输出验证结果
    print("\n" + "="*50)
    print("✅ 旅行受试者数据集验证结果汇总")
    print("="*50)
    if stat_result:
        print(f"1. SSIM变化：均值={stat_result['mean_ssim_diff']:.6f} ± 标准差={stat_result['std_ssim_diff']:.6f}")
        print(f"2. 统计显著性：原始p值={stat_result['raw_p_value']:.6f} | 校正后p值={stat_result['corrected_p_value']:.6f} | 显著={stat_result['is_significant']}")
        print(f"3. 有效受试者数：{stat_result['n_subjects']}")
    print(f"4. 欧氏距离相关性（个体差异保留）：均值={corr_mean:.6f} ± 标准差={corr_std:.6f}")
    
    # 保存结果到文件
    result_data = {
        '指标': [
            'SSIM均值变化', 'SSIM标准差变化', 'Wilcoxon统计量', 
            '原始p值', '校正后p值', '是否显著', '有效受试者数',
            '欧氏距离相关系数均值', '欧氏距离相关系数标准差'
        ],
        '值': [
            stat_result['mean_ssim_diff'] if stat_result else np.nan,
            stat_result['std_ssim_diff'] if stat_result else np.nan,
            stat_result['wilcoxon_stat'] if stat_result else np.nan,
            stat_result['raw_p_value'] if stat_result else np.nan,
            stat_result['corrected_p_value'] if stat_result else np.nan,
            stat_result['is_significant'] if stat_result else np.nan,
            stat_result['n_subjects'] if stat_result else 0,
            corr_mean,
            corr_std
        ]
    }
    result_df = pd.DataFrame(result_data)
    result_csv = os.path.join(Config.OUTPUT_DIR, 'travel_subjects_validation_results.csv')
    result_df.to_csv(result_csv, index=False)
    print(f"验证结果已保存至：{result_csv}")

if __name__ == "__main__":
    main()
