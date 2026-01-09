# 调用系统命令行工具（FSL 的fslreorient2std/flirt/fslroi、N4BiasFieldCorrection）
from subprocess import run

from tempfile import gettempdir, mkdtemp
from uuid import uuid4

import nibabel as nib

import numpy as np
from harmonization.model_architectures import Generator
from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices

# 调用 HD-BET 工具提取脑实质（去除颅骨 / 非脑组织）
from HD_BET.run import run_hd_bet

from os import remove, symlink
from shutil import rmtree

from multiprocessing import Pool  #	多进程池，并行处理

import shutil

# GPU 加速优化：检测到 GPU 时，启用 TensorFlow 混合精度（mixed_float16），提升模型推理速度、降低显存占用；
if len(list_physical_devices('GPU'))>0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    

TMP_DIR = gettempdir()
TEMPLATE_PATH = 'preprocessing/MNI152_T1_1mm_brain.nii.gz'  # MNI152标准脑模板（配准用）

# 生成唯一临时文件路径
def tmp_unique_path(extension='.nii.gz'):
    return f"{TMP_DIR}/{uuid4().hex}{extension}"

def run_cmd(cmd):
    return run(cmd.split(' '), capture_output=True)

# 计算 MRI 裁剪坐标（核心）
def indices_crop(data):
    # count number of zeros
    # 1. 统计x/y/z轴方向的非零像素分布（去除全零背景）
    d1_1=0
    while d1_1<data.shape[0] and np.count_nonzero(data[d1_1,:,:])==0: d1_1+=1
    d1_2=0
    while d1_2<data.shape[0] and np.count_nonzero(data[-d1_2-1,:,:])==0: d1_2+=1
    d2_1=0
    while d2_1<data.shape[1] and np.count_nonzero(data[:,d2_1,:])==0: d2_1+=1
    d2_2=0
    while d2_2<data.shape[1] and np.count_nonzero(data[:,-d2_2-1,:])==0: d2_2+=1
    d3_1=0
    while d3_1<data.shape[1] and np.count_nonzero(data[:,:,d3_1])==0: d3_1+=1
    d3_2=0
    while d3_2<data.shape[1] and np.count_nonzero(data[:,:,-d3_2-1])==0: d3_2+=1
    
    # determine cropping
    # 2. 根据背景多少确定裁剪尺寸：
    # - 背景多：裁剪到固定尺寸（x:160, y:192, z:160），减少计算量；
    # - 背景少：保留更大尺寸（x:176, y:208, z:176），避免裁掉有效脑区；
    if d1_1+d1_2 >= 22:
        if d1_1<11: xmin = d1_1
        elif d1_2<11: xmin = 182-160-d1_2
        else: xmin = 11
        xsize = 160
    else: xmin, xsize = 3,176
        
    if d2_1+d2_2 >= 26:
        if d2_1<13: ymin = d2_1
        elif d2_2<13: ymin = 218-192-d2_2
        else: ymin = 13
        ysize = 192
    else: ymin, ysize = 5,208
    
    if d3_1+d3_2 >= 22:
        if d3_1<11: zmin = d3_1
        elif d3_2<11: zmin = 182-160-d3_2
        else: zmin = 11
        zsize = 160
    else: zmin, zsize = 3,176
    return xmin, xsize, ymin, ysize, zmin, zsize


def run_singleproc(in_mri, out_mri, hd_bet_cpu, just_preprocess=False):
    reorient_path = tmp_unique_path()
    # 重定向：fslreorient2std 将 MRI 图像统一为标准空间方向，避免后续配准偏差
    print(f"run_cmd: fslreorient2std {in_mri} {reorient_path}")
    returned = run_cmd(f"fslreorient2std {in_mri} {reorient_path}")
    if returned.stderr:
        print(f"Problem fslreorient2std : {returned.stderr.decode('utf-8')}")
        return False
    
    # 脑提取：run_hd_bet 提取脑实质（去除颅骨、头皮等非脑组织），生成脑图像 + 脑 mask
    brain_native = tmp_unique_path()
    if hd_bet_cpu: run_hd_bet(reorient_path, brain_native, bet=True, device='cpu', mode='fast', do_tta=False)
    else: run_hd_bet(reorient_path, brain_native, bet=True)
    mask_native = brain_native[:-7]+'_mask.nii.gz'
    
    #  N4 偏置场校正  N4BiasFieldCorrection  消除 MRI 磁场不均匀导致的亮度偏差（提升图像质量）
    n4native = tmp_unique_path() 
    returned = run_cmd(f"N4BiasFieldCorrection -i {brain_native} -x {mask_native} -o {n4native}")
    if returned.returncode != 0:
        print("Problem with N4BiasFieldCorrection")
        return False
    #配准到 MNI 模板 flirt （FSL，6 自由度线性配准）将图像对齐到 MNI152 标准脑空间（跨样本统一空间）；mask 同步配准（最近邻插值）
    n4mni = tmp_unique_path()
    mni_mat = tmp_unique_path('.mat')
    returned = run_cmd(f"flirt -in {n4native} -ref {TEMPLATE_PATH} -omat {mni_mat} -interp trilinear -dof 6 -out {n4mni}")
    if returned.stderr:
        print(f"Problem flirt : {returned.stderr.decode('utf-8')}")
        return False
    # mask 同步配准（最近邻插值）
    mask_mni = tmp_unique_path()
    returned = run_cmd(f"flirt -in {mask_native} -ref {TEMPLATE_PATH} -out {mask_mni} -init {mni_mat} -applyxfm -interp nearestneighbour")
    if returned.stderr:
        print(f"Problem flirt : {returned.stderr.decode('utf-8')}")
        return False
    
    # 中位数归一化：基于脑 mask 计算像素中位数，图像 / 中位数。统一脑区亮度强度（消除扫描设备的强度差异）
    median_mni = tmp_unique_path()
    mri = nib.load(n4mni)
    data = mri.get_fdata()
    mask = nib.load(mask_mni).get_fdata() > 0
    med = np.median(data[mask])
    data = data/med
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, median_mni)
    
    # 裁剪 fslroi（FSL）+ indices_crop，去除背景，仅保留有效脑区
    xmin, xsize, ymin, ysize, zmin, zsize = indices_crop(data)
    median_crop = tmp_unique_path()
    returned = run_cmd(f"fslroi {median_mni} {median_crop} {xmin} {xsize} {ymin} {ysize} {zmin} {zsize}")
    if returned.stderr:
        print(f"Problem fslroi : {returned.stderr.decode('utf-8')}")
        return False
    

    #todo 新增 ========== 新增：保存预处理后文件（同质化前） ==========
    # 读取裁剪后的预处理文件，还原偏移（data-1前的原始值），保存为可直接对比的格式
    preproc_mri_data = nib.load(median_crop).get_fdata()  # 中位数归一化+裁剪后的数据（未减1）
    preproc_mri_img = nib.Nifti1Image(
        preproc_mri_data, 
        affine=mri.affine,  # 复用MNI仿射，保证空间对齐
        header=mri.header
    )
    preproc_mri=out_mri.replace("iguane.nii","preprocessed.nii")
    nib.save(preproc_mri_img, preproc_mri)
    print(f"预处理后文件已保存：{preproc_mri}")
    

    if not just_preprocess:
        # IGUANe harmonization， 加载 Generator 模型推理，后处理（缩放、mask 置 0）
        # 消除不同扫描中心 / 设备的图像差异（核心步骤）

        # 加载预训练同质化模型
        generator = Generator()
        # generator.load_weights('harmonization/iguane_weights.h5')
        gen_path="harmonization/iguane_weights.h5"
        # gen_path="harmonization/my_train_on-harmony/generator_univ.h5"
        # print("推理时采用的模型是我！！ ",gen_path)
        generator.load_weights(gen_path)
        
        
        # 读取裁剪后的预处理 MRI 图像
        mri = nib.load(median_crop)
        data = mri.get_fdata()-1 # 数据偏移 # 中位数 / 均值 减 1 后接近 0
        mask = data>-1  # 生成有效区域mask
        # 总像素数: 4,915,200，有效脑区像素数: 1,598,946 (32.53%)，非脑区像素数: 3,316,254 (67.47%)
        data[~mask] = 0  # 非有效区域置0
        data = np.expand_dims(data, axis=(0,4))  # 扩展维度（batch=1, channel=1，适配模型输入）
        t = convert_to_tensor(data, dtype='float32')  # 转为TensorFlow张量
        
        data = generator(t, training=False).numpy().squeeze()  # 模型推理，挤压维度
        data = (data+1) * 500  # 强度缩放（映射到0~1000）
        data[~mask] = 0 # 非有效区域置0
        data = np.maximum(data, 0)  # 确保无负值
        mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
        nib.save(mri, out_mri)
        print(f"推理后文件已保存：{preproc_mri}")

    # 删除中间临时文件，释放磁盘
    remove(reorient_path)
    remove(brain_native)
    remove(mask_native)
    remove(n4native)
    remove(n4mni)
    remove(mni_mat)
    remove(mask_mni)
    remove(median_mni)
    remove(median_crop)
    
    
    
# 多进程并行：重定向    
def reorient(directory, id_):
    returned = run_cmd(f"fslreorient2std {directory}/base_{id_}.nii.gz {directory}/reorient_{id_}.nii.gz")
    if returned.stderr:
        print(f"Problem fslreorient2std for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False
    return True

# 多进程并行：N4 + 配准 + 归一化 + 裁剪
def n4_flirt_median_crop(directory, id_):
    returned = run_cmd(f"N4BiasFieldCorrection -i {directory}/brainNative_{id_}.nii.gz -x {directory}/brainNative_{id_}_mask.nii.gz -o {directory}/n4native_{id_}.nii.gz")
    if returned.returncode!=0:
        print(f"Problem N4BiasField for entry number {id_}")
        return False
    returned = run_cmd(f"flirt -in {directory}/n4native_{id_}.nii.gz -ref {TEMPLATE_PATH} -omat {directory}/flirtMat_{id_}.mat -interp trilinear -dof 6 -out {directory}/n4mni_{id_}.nii.gz")
    if returned.stderr:
        print(f"Problem flirt for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False
    returned = run_cmd(f"flirt -in {directory}/brainNative_{id_}_mask.nii.gz -ref {TEMPLATE_PATH} -out {directory}/maskMni_{id_}.nii.gz -init {directory}/flirtMat_{id_}.mat -applyxfm -interp nearestneighbour")
    if returned.stderr:
        print(f"Problem flirt apply transform for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False

    mri = nib.load(f"{directory}/n4mni_{id_}.nii.gz")
    data = mri.get_fdata()
    mask = nib.load(f"{directory}/maskMni_{id_}.nii.gz").get_fdata() > 0
    median = np.median(data[mask])
    data = data/median
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, f"{directory}/medianNorm_{id_}.nii.gz")

    xmin, xsize, ymin, ysize, zmin, zsize = indices_crop(data)
    returned = run_cmd(f"fslroi {directory}/medianNorm_{id_}.nii.gz {directory}/crop_{id_}.nii.gz {xmin} {xsize} {ymin} {ysize} {zmin} {zsize}")
    if returned.stderr:
        print(f"Problem fslroi apply transform for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False
    return True
    
def run_multiproc(in_paths, out_paths, n_procs, hd_bet_cpu):
    tmp_dir = mkdtemp()
    ids = range(1, len(in_paths)+1)
    bases = [f"{tmp_dir}/base_{id_}.nii.gz" for id_ in ids]
    for i,id_ in enumerate(ids): symlink(in_paths[i], f"{tmp_dir}/base_{id_}.nii.gz")
    
    print("reorientation of MR images...")
    with Pool(n_procs) as pool:
        flags = pool.starmap(reorient, [(tmp_dir,id_) for id_ in ids])
    ids = [ids[i] for i in range(len(ids)) if flags[i]]
    
    print('\nHD-BET brain extractions...')
    hd_bet_in = [f"{tmp_dir}/reorient_{id_}.nii.gz" for id_ in ids]
    hd_bet_out = [f"{tmp_dir}/brainNative_{id_}.nii.gz" for id_ in ids]
    if hd_bet_cpu: run_hd_bet(hd_bet_in, hd_bet_out, bet=True, device='cpu', mode='fast', do_tta=False)
    else: run_hd_bet(hd_bet_in, hd_bet_out, bet=True)
    
    print('\nBias field correction, registration, median normalization and cropping...')
    print("我有几个核？",n_procs)
    with Pool(n_procs) as pool:
        flags = pool.starmap(n4_flirt_median_crop, [(tmp_dir, id_) for id_ in ids])
    ids = [ids[i] for i in range(len(ids)) if flags[i]]
    
    skip_harmonization =True
    # 如果跳过 harmonization，直接保存预处理结果
    if skip_harmonization:
        print('Saving pre-processed images (no harmonization)...')
        for i, id_ in enumerate(ids, start=1):
            src = f"{tmp_dir}/crop_{id_}.nii.gz"
            dst = out_paths[id_ - 1]
            shutil.copy2(src, dst)          # 也可以用 os.rename 如果同一文件系统
        print('Deletion of temporary files...')
        rmtree(tmp_dir)
        print("跳过 harmonization，直接保存预处理结果")
        return   # 提前结束，后面 harmonization 不跑
    
    print('IGUANe harmonization...')
    generator = Generator()
    generator.load_weights('harmonization/iguane_weights.h5')
    for i,id_ in enumerate(ids, start=1):
        print(f"\tHarmonizing image {i}/{len(ids)}...", end='\r')
        mri = nib.load(f"{tmp_dir}/crop_{id_}.nii.gz")
        data = mri.get_fdata()-1
        mask = data>-1
        data[~mask] = 0
        data = np.expand_dims(data, axis=(0,4))
        t = convert_to_tensor(data, dtype='float32')
        data = generator(t, training=False).numpy().squeeze()
        data = (data+1) * 500
        data[~mask] = 0
        data = np.maximum(data, 0)
        mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
        try: nib.save(mri, out_paths[id_-1].replace("preprocessed","iguane"))
        except FileNotFoundError:
            print(f"Problem saving {out_paths[id_-1].replace('preprocessed','iguane')}")
        
    print('Deletion of temporary files...                     ')
    rmtree(tmp_dir)