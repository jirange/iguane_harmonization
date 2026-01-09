import sys
import getopt
import pandas as pd
import os
from time import time
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
def usage():
    cmd = 'python all_in_one_cal_ssim.py [options]'
    options = [
        ['--csv <input>', 'path to CSV file containing image pairs (columns: a_mri, b_mri)'],
        ['--a-mri <input>', 'path to first MRI image (.nii/.nii.gz) - single pair mode'],
        ['--b-mri <input>', 'path to second MRI image (.nii/.nii.gz) - single pair mode'],
        ['--output <path>', 'output CSV path for batch results (default: ssim_results.csv)'],
        ['--help', 'displays this help']
    ]
    print(f"Usage:\n  {cmd}")
    print('\nAvailable options:')
    for opt, doc in options:
        print(f"\t{opt}{(20-len(opt))*' '}{doc}")
    print('\nNote: Use either --csv for batch mode OR --a-mri/--b-mri for single pair mode')

def calculate_ssim(img1_path, img2_path):
    """
    计算两张NIfTI格式MRI图像的SSIM值（取中间切片计算）
    :param img1_path: 原始MRI图像路径 (.nii/.nii.gz)
    :param img2_path: 处理后MRI图像路径 (.nii/.nii.gz)
    :return: SSIM值
    """
    try:
        # 读取NIfTI图像
        img1_nii = nib.load(img1_path)
        img2_nii = nib.load(img2_path)
        
        # 转换为numpy数组
        img1_data = img1_nii.get_fdata()
        img2_data = img2_nii.get_fdata()
        
        # 确保尺寸一致
        if img1_data.shape != img2_data.shape:
            # 简单的尺寸匹配（取最小维度）
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(img1_data.shape, img2_data.shape))
            img1_data = img1_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            img2_data = img2_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # 取中间切片计算SSIM（3D MRI取轴向中间层）
        mid_slice = img1_data.shape[2] // 2
        img1_slice = img1_data[:, :, mid_slice]
        img2_slice = img2_data[:, :, mid_slice]
        
        # 归一化到0-255（方便SSIM计算）
        img1_slice = (img1_slice - img1_slice.min()) / (img1_slice.max() - img1_slice.min()) * 255
        img2_slice = (img2_slice - img2_slice.min()) / (img2_slice.max() - img2_slice.min()) * 255
        
        # 转换为uint8格式
        img1_slice = img1_slice.astype(np.uint8)
        img2_slice = img2_slice.astype(np.uint8)
        
        # 计算SSIM
        ssim_score = ssim(img1_slice, img2_slice, data_range=img2_slice.max() - img2_slice.min())
        return ssim_score
    
    except Exception as e:
        print(f"Error processing {img1_path} and {img2_path}: {str(e)}")
        return None

def process_batch_csv(csv_path, output_path,iguane_not_preprocessed=False):
    """
    批量处理CSV文件中的图像对
    :param csv_path: CSV文件路径
    :param output_path: 输出结果路径
    :return: SSIM结果列表
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # 检查必要的列
    required_columns = ['a_mri', 'b_mri']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        print(f"Found columns: {list(df.columns)}")
        return None
    
    # 存储结果
    results = []
    ssim_scores = []
    
    print(f"Processing {len(df)} image pairs from {csv_path}")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        a_path = row['a_mri']
        b_path = row['b_mri']
        if iguane_not_preprocessed:
            a_path = a_path.replace("preprocessed","iguane")
            b_path = b_path.replace("preprocessed","iguane")
        
        print(f"Processing pair {idx + 1}/{len(df)}:")
        print(f"  A: {a_path}")
        print(f"  B: {b_path}")
        
        # 检查文件是否存在
        if not os.path.exists(a_path):
            print(f"  Warning: File not found - {a_path}")
            ssim_score = None
        elif not os.path.exists(b_path):
            print(f"  Warning: File not found - {b_path}")
            ssim_score = None
        else:
            # 计算SSIM
            ssim_score = calculate_ssim(a_path, b_path)
            if ssim_score is not None:
                ssim_scores.append(ssim_score)
        
        # 存储结果
        result_row = row.copy()
        result_row['ssim'] = ssim_score
        results.append(result_row)
        
        if ssim_score is not None:
            print(f"  SSIM: {ssim_score:.4f}")
        else:
            print(f"  SSIM: Failed")
        print("-" * 40)
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # 计算统计信息
    if ssim_scores:
        stats = {
            'count': len(ssim_scores),
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores),
            'min': np.min(ssim_scores),
            'max': np.max(ssim_scores),
            'median': np.median(ssim_scores)
        }
        
        print("\n" + "="*60)
        print("SSIM Statistics:")
        print("="*60)
        print(f"Total successful pairs: {stats['count']}")
        print(f"Mean SSIM: {stats['mean']:.4f}")
        print(f"Standard deviation: {stats['std']:.4f}")
        print(f"Min SSIM: {stats['min']:.4f}")
        print(f"Max SSIM: {stats['max']:.4f}")
        print(f"Median SSIM: {stats['median']:.4f}")
        print("="*60)

        # 1. 构造一行 DataFrame
        subject_id = df['subject_id'].iloc[0]
        if subject_id != df['subject_id'].iloc[-1]:
            print("出问题了，这不是同一个sub的！！！！！")
        record = {
            'subject_id': subject_id,
            'count':      stats['count'],
            'mean':       stats['mean'],
            'std':        stats['std'],
            'min':        stats['min'],
            'max':        stats['max'],
            'median':     stats['median']
        }
        df_stat = pd.DataFrame([record])

        # 2. 保存路径
        stat_file = Path("/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/all_sub_ssim_stats.csv")
        if iguane_not_preprocessed:
            stat_file = Path("/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/all_sub_ssim_stats_after_iguane.csv")

        # 3. 追加模式：文件存在就追加，不存在就写表头
        header = not stat_file.exists()
        df_stat.to_csv(stat_file, mode='a', header=header, index=False)
        print("统计信息已经保存到  ",stat_file)
        
        
        return results, stats
    else:
        print("\nNo successful SSIM calculations!")
        return results, None

def main():
    # 解析命令行参数
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'h', ['help', 'csv=', 'a-mri=', 'b-mri=', 'output='])
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        usage()
        sys.exit(2)
    
    csv_file = None
    a_mri = None
    b_mri = None
    output_file = 'ssim_results.csv'
    
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt == '--csv':
            csv_file = arg
        elif opt == '--a-mri':
            a_mri = arg
        elif opt == '--b-mri':
            b_mri = arg
        elif opt == '--output':
            output_file = arg
    
    # 检查参数模式
    if csv_file:
        # 批量模式
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found - {csv_file}")
            sys.exit(1)
        
        sub_ids=['sub-03997', 'sub-14230', 'sub-16841', 'sub-14482', 'sub-16794', 'sub-13192', 'sub-16975', 'sub-16974', 'sub-14229', 'sub-10975', 'sub-16766', 'sub-16981', 'sub-03286', 'sub-14221', 'sub-16793', 'sub-16745', 'sub-13305', 'sub-16842', 'sub-15320', 'sub-12813']
        for sub in sub_ids:
            csv_file = f"data/ON-Harmony/{sub}_t1_pairs.csv"
            print("接下来处理的是 ",csv_file)
            results, stats = process_batch_csv(csv_file, output_file)
            # results, stats = process_batch_csv(csv_file, output_file,iguane_not_preprocessed=True)

        # results, stats = process_batch_csv(csv_file, output_file)

        if results is None:
            sys.exit(1)
            
    elif a_mri and b_mri:
        # 单对模式
        if not os.path.exists(a_mri):
            print(f"Error: File not found - {a_mri}")
            sys.exit(1)
        if not os.path.exists(b_mri):
            print(f"Error: File not found - {b_mri}")
            sys.exit(1)
        
        try:
            ssim_score = calculate_ssim(a_mri, b_mri)
            print("="*50)
            print(f"SSIM between {a_mri} and {b_mri}: {ssim_score:.4f}")
            print("="*50)
            print("Note: SSIM ranges from 0 to 1 (1 = perfect similarity)")
        except Exception as e:
            print(f"\nError calculating SSIM: {e}")
            print("Please check:")
            print("1. Image paths are correct")
            print("2. Images are valid NIfTI files (.nii/.nii.gz)")
            sys.exit(1)
    else:
        print("Error: Either --csv for batch mode OR --a-mri/--b-mri for single pair mode is required!")
        usage()
        sys.exit(2)
        
if __name__ == "__main__":
    t_start = time()
    main()
    print(f"\nEnd of execution, total processing time = {time()-t_start:.0f} seconds")