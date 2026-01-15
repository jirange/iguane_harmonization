# 适配目录结构的 NIfTI 转 TFRecord 脚本
# SALD 是 BIDS 标准格式，需递归进入sub-xxx/anat/目录提取*_T1w.nii.gz；IXI 是平铺格式，直接提取*_T1.nii.gz。以下是完整转换脚本：


import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from input_pipeline.constants import IMAGE_SHAPE  # 你的MRI尺寸，如(182,218,182,1)
import re
import pandas as pd
from collections import defaultdict

# ===================== 旅行者：按 session 拆分 =====================
def split_traveler_by_session(bids_root):
    """
    递归 BIDS，按 session 名拆分 T1w
    :return: dict {session: [nii.gz, ...]}
    """
    from collections import defaultdict
    sess2files = defaultdict(list)
    print(os.listdir(bids_root))
    for sub in os.listdir(bids_root):
        sub_path = os.path.join(bids_root, sub)
        if not os.path.isdir(sub_path) or not sub.startswith('sub-'): 
            continue
        for ses in os.listdir(sub_path):
            ses_path = os.path.join(sub_path, ses)
            if not os.path.isdir(ses_path) or not ses.startswith('ses-'): 
                continue
            anat_dir = os.path.join(ses_path, 'anat')
            if not os.path.exists(anat_dir): 
                continue
            for f in os.listdir(anat_dir):
                if f.endswith('_T1w.nii.gz'):
                    sess2files[ses].append(os.path.join(anat_dir, f))
    # 打印
    for ses, files in list(sess2files.items()):      # 用 list 复制，避免边迭代边删
        n = len(files)
        print(f'{ses}: {n} 张 T1w', end='')
        if n < 10:
            del sess2files[ses]                      # 删掉不足 10 张的 session
            print('  → 已剔除')
        else:
            print()
    return sess2files

def split_traveler_on_preprocessed(bids_root):
    """
    递归 BIDS，按 session 名拆分 T1w
    :return: dict {session: [nii.gz, ...]}
    """
    print("递归 BIDS，按 session 名拆分 T1w")

    from collections import defaultdict
    sess2files = defaultdict(list)
    # print(os.listdir(bids_root))

    for f in os.listdir(bids_root):
        if f.endswith('.nii.gz') and "sub" in f and "ses" in f:
            # 用正则抓 sub-<label> 和 ses-<label>
            sub_match = re.search(r'(sub-[a-zA-Z0-9]+)', f)
            ses_match = re.search(r'(ses-[a-zA-Z0-9]+)', f)
            subject_id = sub_match.group(1) if sub_match else None
            session_id = ses_match.group(1) if ses_match else None
            sess2files[session_id].append(os.path.join(bids_root, f))
        elif f.endswith('.nii.gz') and "sub" in f and "ses" not in f:
            sess2files["Unkonw"].append(os.path.join(bids_root, f))

    # 打印
    for ses, files in list(sess2files.items()):      # 用 list 复制，避免边迭代边删
        n = len(files)
        print(f'{ses}: {n} 张 T1w', end='')
        if n < 10:
            del sess2files[ses]                      # 删掉不足 10 张的 session
            print('  → 已剔除')
        else:
            print()
    return sess2files

def split_traveler_on_preprocessed_by_table(bids_root, participants_tsv="/home/lengjingcheng/datasets/ABIDE/ABIDE2_BIDS/participants.tsv"):
    """
    按 participants.tsv 中的 site_id 拆分 T1w
    :param bids_root:  预处理输出目录（仅含 nii.gz）
    :param participants_tsv:  BIDS 根目录下的 participants.tsv 全路径
    :return: dict {site_id: [nii.gz, ...]}
    """
    print("按 participants.tsv 中的 site_id 拆分")
    # 1. 读 TSV，建立 sub -> site 映射
    df = pd.read_csv(participants_tsv, sep='\t')
    sub2site = dict(zip(df['participant_id'], df['site_id']))   # sub-29006 -> ABIDEII-BNI_1

    # 2. 遍历目录，按站点收文件
    site2files = defaultdict(list)
    for f in os.listdir(bids_root):
        if f.endswith('.nii.gz'):
            sub_match = re.search(r'(sub-[a-zA-Z0-9]+)', f)
            if not sub_match:
                continue
            subject_id = sub_match.group(1)
            site_id = sub2site.get(subject_id)        # 查表
            if site_id is None:                       # 防 KeyError
                print(f'Warning: {subject_id} not found in TSV, skip')
                continue
            site2files[site_id].append(os.path.join(bids_root, f))

    # 3. 同样的“不足 10 张就剔除”规则
    for site, files in list(site2files.items()):
        n = len(files)
        print(f'{site}: {n} 张 T1w', end='')
        if n < 10:
            del site2files[site]
            print('  → 已剔除')
        else:
            print()
    return site2files

def batch_convert_traveler(bids_root, out_dir, ref_session=None, participants_tsv=None,name=None):
    """
    一次性生成 6 个 TFRecord，并把 ref_session 单独命名方便后面直接引用
    """
    os.makedirs(out_dir, exist_ok=True)
    if "preprocessed" in bids_root:
        if not participants_tsv:
            sess2files = split_traveler_on_preprocessed(bids_root)
        else:
            sess2files = split_traveler_on_preprocessed_by_table(bids_root,participants_tsv=participants_tsv)

    else:
        sess2files = split_traveler_by_session(bids_root)

    
    # 写参考域
    if ref_session:
        ref_tfr = os.path.join(out_dir, f'{name}_{ref_session}.records.gz')
        write_nifti_to_tfrecord(sess2files[ref_session], ref_tfr)

    # 写 5 个源域
    for ses, files in sess2files.items():
        if ses == ref_session: 
            continue
        src_tfr = os.path.join(out_dir, f'{name}_{ses}_preprocessed.records.gz')
        write_nifti_to_tfrecord(files, src_tfr)



# ===================== 新增：IXI医院拆分逻辑 =====================
def split_ixi_by_hospital(ixi_root):
    """
    按文件名拆分IXI为Guys、HH、IOP三家医院
    :param ixi_root: IXI根目录
    :return: 字典{医院名: NIfTI文件列表}
    """
    ixi_hospitals = {
        "Guys": [],
        "HH": [],
        "IOP": []
    }
    # 遍历IXI所有T1文件
    for file in os.listdir(ixi_root):
        print(file)
        if not file.endswith(".nii.gz"):
            continue
        file_path = os.path.join(ixi_root, file)
        # 按文件名判断医院（优先判断Guys，避免字符重叠）
        if "Guys" in file:
            ixi_hospitals["Guys"].append(file_path)
        elif "HH" in file:
            ixi_hospitals["HH"].append(file_path)
        elif "IOP" in file:
            ixi_hospitals["IOP"].append(file_path)
        else:
            print(f"跳过未知医院的文件：{file}")
    
    # 打印拆分结果
    for hospital, files in ixi_hospitals.items():
        print(f"IXI-{hospital}医院：找到{len(files)}个T1加权像")
    return ixi_hospitals

# ===================== 复用：SALD文件查找逻辑 =====================
def find_sald_t1_files(sald_root):
    """查找SALD（BIDS格式）的T1w文件"""
    sald_t1_files = []
    for sub_dir in os.listdir(sald_root):
        sub_path = os.path.join(sald_root, sub_dir)
        if not os.path.isdir(sub_path) or not sub_dir.startswith("sub-"):
            continue
        anat_dir = os.path.join(sub_path, "anat")
        if not os.path.exists(anat_dir):
            continue
        for file in os.listdir(anat_dir):
            if file.endswith("_T1w.nii.gz"):
                sald_t1_files.append(os.path.join(anat_dir, file))
    print(f"SALD目标域：找到{len(sald_t1_files)}个T1加权像")
    return sald_t1_files
# ===================== 复用：TFRecord写入逻辑 =====================
# def write_nifti_to_tfrecord(nifti_files, output_tfr, compression_type='GZIP'):
#     """将NIfTI列表写入单个TFRecord"""
#     writer = tf.io.TFRecordWriter(
#         output_tfr,
#         options=tf.io.TFRecordOptions(compression_type=compression_type)
#     )
    
#     for idx, nifti_path in enumerate(nifti_files):
#         try:
#             # 读取并适配形状
#             img = nib.load(nifti_path)
#             mri_data = img.get_fdata()
#             mri_data = np.expand_dims(mri_data, axis=-1).astype(np.float32)
#             if mri_data.shape != IMAGE_SHAPE:
#                 mri_data = np.resize(mri_data, IMAGE_SHAPE)
            
#             # 构建Feature和Example
#             feature = {
#                 'mri': tf.train.Feature(
#                     float_list=tf.train.FloatList(value=mri_data.flatten())
#                 )
#             }
#             example = tf.train.Example(features=tf.train.Features(feature=feature))
#             writer.write(example.SerializeToString())
            
#             if (idx+1) % 50 == 0:
#                 print(f"已转换{idx+1}/{len(nifti_files)}个文件")
#         except Exception as e:
#             print(f"跳过损坏文件{nifti_path}：{e}")
    
#     writer.close()
#     print(f"✅ TFRecord保存完成：{output_tfr}")

def write_nifti_to_tfrecord(nifti_files, output_tfr, compression_type='GZIP', split_at=600):
    """将NIfTI列表写入单个TFRecord，超过split_at个则自动分卷"""
    for chunk_id, start in enumerate(range(0, len(nifti_files), split_at), 1):
        chunk = nifti_files[start:start + split_at]
        out_file = output_tfr.replace('.tfrecord', f'_part{chunk_id}.tfrecord')
        writer = tf.io.TFRecordWriter(out_file,
                                      options=tf.io.TFRecordOptions(compression_type=compression_type))
        for idx, nifti_path in enumerate(chunk, start + 1):
            try:
                img = nib.load(nifti_path)
                mri_data = img.get_fdata()
                mri_data = np.expand_dims(mri_data, axis=-1).astype(np.float32)
                if mri_data.shape != IMAGE_SHAPE:
                    mri_data = np.resize(mri_data, IMAGE_SHAPE)
                feature = {'mri': tf.train.Feature(
                    float_list=tf.train.FloatList(value=mri_data.flatten()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                if idx % 50 == 0:
                    print(f"已转换{idx}/{len(nifti_files)}个文件")
            except Exception as e:
                print(f"跳过损坏文件{nifti_path}：{e}")
        writer.close()
        print(f"✅ TFRecord保存完成：{out_file}")


# ===================== 批量转换主函数 =====================
def batch_convert_datasets(sald_root, ixi_root, output_tfr_dir):
    """批量转换SALD（目标域）和IXI（分医院源域）为TFRecord"""
    os.makedirs(output_tfr_dir, exist_ok=True)
    
    # 1. 转换SALD（目标域，统一为一个TFRecord）
    sald_t1_files = find_sald_t1_files(sald_root)
    sald_tfr = os.path.join(output_tfr_dir, "sald_target.records.gz")
    write_nifti_to_tfrecord(sald_t1_files, sald_tfr)
    
    # 2. 拆分并转换IXI（分三家医院作为源域）
    ixi_hospitals = split_ixi_by_hospital(ixi_root)
    for hospital, files in ixi_hospitals.items():
        if not files:
            continue
        ixi_tfr = os.path.join(output_tfr_dir, f"ixi_source_{hospital}.records.gz")
        write_nifti_to_tfrecord(files, ixi_tfr)
# ===================== 调用示例（替换为你的实际路径） =====================
if __name__ == "__main__":
    # # 你的数据集根目录
    # sald_root = "/home/lengjingcheng/datasets/SALD/sub-031274_sub-031767"  # SALD根目录（包含sub-031766、sub-031767等）
    # ixi_root = "/home/lengjingcheng/datasets/IXI"          # IXI根目录（平铺T1文件）
    # output_tfr_dir = "/home/lengjingcheng/datasets/tfrecords"  # TFRecord输出目录
    
    # # 执行转换

    # """批量转换SALD（目标域）和IXI（分医院源域）为TFRecord"""
    # os.makedirs(output_tfr_dir, exist_ok=True)
    
    # # # 1. 转换SALD（目标域，统一为一个TFRecord）
    # # sald_t1_files = find_sald_t1_files(sald_root)
    # # sald_tfr = os.path.join(output_tfr_dir, "sald_target.records.gz")
    # # write_nifti_to_tfrecord(sald_t1_files, sald_tfr)
    
    # # 2. 拆分并转换IXI（分三家医院作为源域）
    # ixi_hospitals = split_ixi_by_hospital(ixi_root)
    # for hospital, files in ixi_hospitals.items():
    #     if not files:
    #         continue
    #     ixi_tfr = os.path.join(output_tfr_dir, f"ixi_source_{hospital}.records.gz")
    #     write_nifti_to_tfrecord(files, ixi_tfr)


    # ===================== 调用 =====================
    # bids_root = '/home/lengjingcheng/datasets/ON-Harmony/ds004712-download'  # 旅行者根目录
    # out_dir   = '/home/lengjingcheng/datasets/tfrecords'
    # batch_convert_traveler(bids_root, out_dir, ref_session='ses-NOT1ACH001')

    # ===================== ABIDE2 =====================
    # bids_root = '/home/lengjingcheng/codes/iguane_harmonization/data/ABIDE2/preprocessed'  # 旅行者根目录
    out_dir   = '/home/lengjingcheng/datasets/tfrecords'
    # # batch_convert_traveler(bids_root, out_dir, ref_session='ses-1')
    
    # batch_convert_traveler(bids_root, out_dir,participants_tsv="/home/lengjingcheng/datasets/ABIDE/ABIDE2_BIDS/participants.tsv")

    # ===================== IXI =====================
    # # 2. 拆分并转换IXI（分三家医院作为源域）
    ixi_hospitals = split_ixi_by_hospital(ixi_root="/home/lengjingcheng/codes/iguane_harmonization/data/IXI/preprocessed")
    for hospital, files in ixi_hospitals.items():
        if not files:
            continue
        ixi_tfr = os.path.join(out_dir, f"ixi_{hospital}_preprocessed.records.gz")
        write_nifti_to_tfrecord(files, ixi_tfr)

    # ===================== SALD =====================
    # 1. 转换SALD（目标域，统一为一个TFRecord）
    # batch_convert_traveler("/home/lengjingcheng/codes/iguane_harmonization/data/SALD/preprocessed", out_dir,name="SALD")

    
