import os
import pandas as pd
import glob
import argparse
from pathlib import Path

from tqdm import tqdm
import re
from pathlib import Path
def find_t1_images(base_path, subject_id):
    """
    查找指定被试的所有T1图像
    :param base_path: 基础路径，如 ~/datasets/ON-Harmony/ds004712-download
    :param subject_id: 被试ID，如 sub-03286
    :return: 包含T1图像信息的列表
    """
    # 展开用户路径
    base_path = os.path.expanduser(base_path)
    
    # 构建被试路径
    if "IXI" in base_path:
        subject_path = base_path
    else:
        subject_path = os.path.join(base_path, subject_id)
    
    if not os.path.exists(subject_path):
        print(f"Error: Subject path not found - find_t1_images-{subject_path}")
        return []
    
    # 查找所有T1w.nii.gz文件
    if "SALD" in base_path:
        t1_pattern = os.path.join(subject_path, "anat/*_T1w.nii.gz")
    elif "IXI" in base_path:
        t1_pattern = os.path.join(subject_path, f"{subject_id}-*-T1.nii.gz")
        print(t1_pattern)
    else:
        t1_pattern = os.path.join(subject_path, "ses-*/anat/*_T1w.nii.gz")
    
    t1_files = glob.glob(t1_pattern)
    
    t1_images = []
    
    for t1_file in t1_files:
        # 解析会话信息
        path_parts = Path(t1_file).parts
        session_dir = [part for part in path_parts if part.startswith('ses-')]
        
        if session_dir:
            session_id = session_dir[0]  # 如 ses-NOT1ACH001
            
            # 提取扫描仪信息（从会话名）
            scanner_info = session_id.replace('ses-', '')
            
            t1_images.append({
                'file_path': t1_file,
                'subject_id': subject_id,
                'session_id': session_id,
                'scanner_info': scanner_info,
                'filename': os.path.basename(t1_file)
            })
        else:
            # print("没有 扫描仪目录session_dir  没有扫描仪信息 ")
            if "IXI" in base_path:
                fname = os.path.basename(t1_file)
                pid, site = re.match(r'IXI(\d+)-(\w+)-', fname).groups()
                session_id = site
                scanner_info = {"HH": "Philips 3T",
                                "Guys": "Philips 1.5T",
                                "IOP": "GE 1.5T"}.get(site, "Unknown")
            else:
                session_id = scanner_info = None
            t1_images.append({
                'file_path': t1_file,
                'subject_id': subject_id,
                'session_id': session_id,
                'scanner_info': scanner_info,
                'filename': os.path.basename(t1_file)
            })
    
    return t1_images

def find_t1_images4pairs(base_path, subject_id):
    """
    查找指定被试的所有T1图像
    :param base_path: 基础路径，如 codes/iguane_harmonization/data/ON-Harmony/preprocessed
    :param subject_id: 被试ID，如 sub-03286
    :return: 包含T1图像信息的列表
    """
    # 展开用户路径
    base_path = os.path.expanduser(base_path)
    
    # 构建被试路径
    # subject_path = os.path.join(base_path, subject_id)
    
    if not os.path.exists(base_path):
        print(f"Error: Subject path not found - {base_path}")
        return []
    
    # 查找所有T1w.nii.gz文件
    # t1_pattern = os.path.join(subject_path, "ses-*/anat/*_T1w.nii.gz")
    t1_pattern = os.path.join(base_path, f"{subject_id}*_T1w-preprocessed.nii.gz")
    t1_files = glob.glob(t1_pattern)
    print(t1_files)
    
    t1_images = []
    
    for t1_file in t1_files:
        # 解析会话信息
        path_parts = Path(t1_file).parts
        # print("path_parts: ",path_parts)
        fname = path_parts[-1]
            # 先去掉可能的后缀 .nii.gz 或 .nii
        if fname.endswith('.nii.gz'):
            fname = fname[:-7]
        elif fname.endswith('.nii'):
            fname = fname[:-4]

        # 用正则抓 sub-<label> 和 ses-<label>
        sub_match = re.search(r'(sub-[a-zA-Z0-9]+)', fname)
        ses_match = re.search(r'(ses-[a-zA-Z0-9]+)', fname)

        subject_id = sub_match.group(1) if sub_match else None
        session_id = ses_match.group(1) if ses_match else None     
            
        t1_images.append({
            'file_path': t1_file,
            'subject_id': subject_id,
            'session_id': session_id,
            'filename': os.path.basename(t1_file)
        })
    
    return t1_images

def generate_pairs(t1_images):
    """
    生成T1图像的两两配对（避免重复配对）
    :param t1_images: T1图像信息列表
    :return: 配对列表
    """
    pairs = []
    
    for i in range(len(t1_images)):
        for j in range(i + 1, len(t1_images)):
            img1 = t1_images[i]
            img2 = t1_images[j]
            
            pairs.append({
                'a_mri': img1['file_path'],
                'b_mri': img2['file_path'],
                'subject_id': img1['subject_id'],
                'session_a': img1['session_id'],
                'session_b': img2['session_id'],
                'filename_a': img1['filename'],
                'filename_b': img2['filename']
            })
    
    return pairs

def create_csv_for_subject(base_path, subject_id, output_csv=None):
    """
    为指定被试创建SSIM计算用的CSV文件
    :param base_path: 基础路径
    :param subject_id: 被试ID
    :param output_csv: 输出CSV文件名（可选）
    :return: 配对数量
    """
    print(f"Searching for T1 images for {subject_id}...")
    
    # 查找T1图像
    t1_images = find_t1_images4pairs(base_path, subject_id)
    
    if not t1_images:
        print(f"No T1 images found for {subject_id}")
        return 0
    
    print(f"Found {len(t1_images)} T1 images:")
    for img in t1_images:
        print(f"  - {img['session_id']}: {img['file_path']}")
    
    # 生成配对
    pairs = generate_pairs(t1_images)
    
    if not pairs:
        print("No pairs generated")
        return 0
    
    print(f"\nGenerated {len(pairs)} unique pairs")
    
    # 创建DataFrame
    df = pd.DataFrame(pairs)
    print(output_csv)
    if '.csv' not in output_csv:
        output_csv=os.path.join(output_csv, f"{subject_id}_t1_pairs.csv")# 保存CSV文件
    df.to_csv(output_csv, index=False)
    
    print(f"\nCSV file saved: {output_csv}")
    print(f"Total pairs: {len(pairs)}")
    
    # 显示配对统计
    print("\nPairing summary:")
    scanner_pairs = df.groupby(['session_a', 'session_b']).size().reset_index(name='count')
    for _, row in scanner_pairs.iterrows():
        print(f"  {row['session_a']} vs {row['session_b']}: {row['count']} pairs")
    
    return len(pairs)

def batch_process_multiple_subjects(base_path, subject_list=None, output_dir=None, 
                                  combine_all=True, combined_csv="on-harmony-all_subjects_t1_pairs.csv"):
    """
    批量处理多个被试
    :param base_path: 基础路径
    :param subject_list: 被试ID列表（可选）
    :param output_dir: 输出目录（可选）
    :param combine_all: 是否合并所有被试到一个CSV文件
    :param combined_csv: 合并后的CSV文件名
    """
    if subject_list is None:
        base_path = os.path.expanduser(base_path)
        subject_pattern = os.path.join(base_path, "sub-*")
        subject_dirs = glob.glob(subject_pattern)
        # print(subject_dirs)

        subject_list = [os.path.basename(d) for d in subject_dirs]
        subject_list = [d[:9] for d in subject_list]
        subject_list = list(set(subject_list))

        print(subject_list)

    
    if not subject_list:
        print("No subjects found")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_pairs = []  # 存储所有被试的配对
    
    for subject_id in tqdm(subject_list, desc="Processing subjects"):
        if output_dir:
            output_csv = os.path.join(output_dir, f"{subject_id}_t1_pairs.csv")
        else:
            output_csv = None
        
        pairs_count = create_csv_for_subject(base_path, subject_id, output_csv)
        
        if combine_all and pairs_count > 0:
            # 读取刚生成的CSV文件并添加到总列表
            if output_dir:
                df = pd.read_csv(output_csv)
                all_pairs.append(df)
    
    if combine_all and all_pairs:
        # 合并所有被试的数据
        combined_df = pd.concat(all_pairs, ignore_index=True)
        
        if output_dir:
            combined_path = os.path.join(output_dir, combined_csv)
        else:
            combined_path = combined_csv
            
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined CSV saved: {combined_path}")
        print(f"Total pairs across all subjects: {len(combined_df)}")
    print(subject_list)

def generate_preprocessing_csv(base_path, subject_id, output_csv=None, preprocessing_suffix='-preprocessed'):
    """
    生成用于预处理的CSV文件
    :param base_path: 基础路径
    :param subject_id: 被试ID
    :param output_csv: 输出CSV文件名（可选）
    :param preprocessing_suffix: 预处理后文件的后缀
    :return: 生成的CSV行数
    """
    print(f"Generating preprocessing CSV for {subject_id}...")
    
    # 查找T1图像
    t1_images = find_t1_images(base_path, subject_id)
    
    if not t1_images:
        print(f"No T1 images found for {subject_id}")
        return 0
    
    print(f"Found {len(t1_images)} T1 images:")
    for img in t1_images:
        if "SALD" not in base_path:
            print(f"  - {img['session_id']}: {img['file_path']}")
    
    # 创建预处理数据
    preprocessing_data = []
    
    for img in t1_images:
        original_path = img['file_path']
        
        # 生成预处理后的输出路径
        # 获取文件所在的目录
        file_dir = os.path.dirname(original_path)
        original_filename = os.path.basename(original_path)
        
        # 分割文件名和扩展名
        name_part, ext_part = os.path.splitext(original_filename)
        if ext_part == '.gz':
            # 处理.nii.gz情况
            name_part, ext_part2 = os.path.splitext(name_part)
            ext_part = ext_part2 + ext_part
        
        # 构建预处理后的文件名

        preprocessed_filename = f"{name_part}{preprocessing_suffix}{ext_part}"
        preprocessed_path = os.path.join(os.path.dirname(output_csv),"preprocessed", preprocessed_filename)
        
        if 'session_id' in img:
            preprocessing_data.append({
                'in_mri': original_path,
                'out_mri': preprocessed_path,
                'subject_id': img['subject_id'],
                'session_id': img['session_id'],
                'scanner_info': img['scanner_info'],
                'original_filename': original_filename,
                'preprocessed_filename': preprocessed_filename
            })
        else:
            print("确认一下，无扫描仪信息，是SALD吗")
            preprocessing_data.append({
                'in_mri': original_path,
                'out_mri': preprocessed_path,
                'subject_id': img['subject_id'],
                'original_filename': original_filename,
                'preprocessed_filename': preprocessed_filename
            })    
    # 创建DataFrame
    df = pd.DataFrame(preprocessing_data)
    
    # 如果未指定输出文件名，自动生成
    if output_csv is None:
        output_csv = f"{subject_id}_preprocessing.csv"
    
    # 保存CSV文件
    df.to_csv(output_csv, index=False)
    
    print(f"\nPreprocessing CSV saved: {output_csv}")
    print(f"Total images: {len(preprocessing_data)}")
    
    # 显示统计信息
    print("\nPreprocessing summary:")
    for data in preprocessing_data:
        print(f"  IN:  {data['original_filename']}")
        print(f"  OUT: {data['preprocessed_filename']}")
        print(f"  PATH: {data['in_mri']} -> {data['out_mri']}")
        print("-" * 60)
    
    return len(preprocessing_data)


def batch_generate_preprocessing_csv(base_path, subject_list=None, output_dir=None, 
                                   preprocessing_suffix='-preprocessed', combine_all=True, 
                                   combined_csv="all_subjects_preprocessing.csv"):
    """
    批量生成多个被试的预处理CSV文件
    :param base_path: 基础路径
    :param subject_list: 被试ID列表（可选）
    :param output_dir: 输出目录（可选）
    :param preprocessing_suffix: 预处理后文件的后缀
    :param combine_all: 是否合并所有被试到一个CSV文件
    :param combined_csv: 合并后的CSV文件名
    """
    print("是我是我")

    if subject_list is None:
        base_path = os.path.expanduser(base_path)
        subject_pattern = os.path.join(base_path, "sub-*")
        subject_dirs = glob.glob(subject_pattern)
        subject_list = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]
    
    if not subject_list:
        print("No subjects found")
        subject_pattern = os.path.join(base_path, "*.nii.gz")
        subject_dirs = glob.glob(subject_pattern)
        print(subject_dirs)

        subject_list = [os.path.basename(d) for d in subject_dirs]
        if "IXI" in base_path:
            subject_list = [re.match(r'(IXI\d+)-', f).group(1) for f in subject_list if f.endswith('.nii.gz')]
        else:
            print("出问题了，文件名匹配不对！")
        subject_list = list(set(subject_list))

        print(subject_list)

        # return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_images = []  # 存储所有被试的预处理数据
    
    for subject_id in tqdm(subject_list, desc="Generating preprocessing CSVs"):
        if output_dir:
            output_csv = os.path.join(output_dir, f"{subject_id}_preprocessing.csv")
        else:
            output_csv = None
        images_count = generate_preprocessing_csv(base_path, subject_id, output_csv, preprocessing_suffix)
        print("images_count",images_count,combine_all,output_dir)
        
        if combine_all and images_count > 0:
            # 读取刚生成的CSV文件并添加到总列表
            if output_dir:
                df = pd.read_csv(output_csv)
                all_images.append(df)
    print(combine_all,all_images)
    if combine_all and all_images:
        # 合并所有被试的数据
        combined_df = pd.concat(all_images, ignore_index=True)
        
        if output_dir:
            combined_path = os.path.join(output_dir, combined_csv)
        else:
            combined_path = combined_csv
            
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined preprocessing CSV saved: {combined_path}")
        print(f"Total images across all subjects: {len(combined_df)}")


def main():
    parser = argparse.ArgumentParser(description='Generate CSV files for T1 image processing')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')
    
    # 预处理模式
    prep_parser = subparsers.add_parser('preprocessing', help='Generate preprocessing CSV')
    prep_parser.add_argument('--base-path', type=str, default='~/datasets/ON-Harmony/ds004712-download',
                           help='Base path to the dataset')
    prep_parser.add_argument('--subject', type=str, help='Subject ID (e.g., sub-03286)')
    prep_parser.add_argument('--subject-list', type=str, nargs='+', help='List of subject IDs')
    prep_parser.add_argument('--output', type=str, default="/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony",help='Output CSV file or directory')
    prep_parser.add_argument('--suffix', type=str, default='-preprocessed',
                           help='Suffix for preprocessed files (default: -preprocessed)')
    prep_parser.add_argument('--batch', action='store_true', help='Process all subjects in batch mode')
    
    # 配对模式
    pair_parser = subparsers.add_parser('pairing', help='Generate pairing CSV for SSIM calculation')
    pair_parser.add_argument('--base-path', type=str, default='/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/preprocessed',
                           help='Base path to the dataset')
    pair_parser.add_argument('--subject', type=str, help='Subject ID (e.g., sub-03286)')
    pair_parser.add_argument('--subject-list', type=str, nargs='+', help='List of subject IDs')
    pair_parser.add_argument('--output', type=str, default="/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/",help='Output CSV file or directory')
    pair_parser.add_argument('--batch', action='store_true', help='Process all subjects in batch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocessing':
        # 预处理模式
        if args.subject:
            generate_preprocessing_csv(args.base_path, args.subject, args.output, args.suffix)
        elif args.subject_list:
            batch_generate_preprocessing_csv(args.base_path, args.subject_list, args.output, args.suffix)
        elif args.batch:
            print("args.output",args.output)
            batch_generate_preprocessing_csv(args.base_path, output_dir=args.output, preprocessing_suffix=args.suffix)
            print("args.output",args.output)
        else:
            prep_parser.print_help()
    
    elif args.mode == 'pairing':
        # 配对模式
        if args.subject:
            create_csv_for_subject(args.base_path, args.subject, args.output)
        elif args.subject_list:
            batch_process_multiple_subjects(args.base_path, args.subject_list, args.output)
        elif args.batch:
            batch_process_multiple_subjects(args.base_path, output_dir=args.output)
        else:
            pair_parser.print_help()
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # 生成预处理CSV")
        print("  python generate_t1_pairs.py preprocessing --subject sub-03286")
        print("  python generate_t1_pairs.py preprocessing --subject sub-03286 --output prep.csv")
        print("  python generate_t1_pairs.py preprocessing --batch --output prep_dir/")
        print()
        print("  # 生成配对CSV")
        print("  python generate_t1_pairs.py pairing --subject sub-03286")
        print("  python generate_t1_pairs.py pairing --subject-list sub-03286 sub-03287 --output pair_dir/")

if __name__ == "__main__":
    main()