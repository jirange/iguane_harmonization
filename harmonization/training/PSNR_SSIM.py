#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-D 整卷推理版：NIfTI → 3-D 模型 → PSNR/SSIM
"""
import os, re, sys, argparse, pandas as pd, numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import tensorflow as tf
sys.path.append('..')
from model_architectures import Generator   # 你的 Generator

import numpy as np

# ===================== 仅改这里 =====================
# MODEL_WEIGHTS_PATH = '/home/lengjingcheng/codes/iguane_harmonization/harmonization/my_train_sald-ixi2-abide2/latest_genUniv.h5'
MODEL_WEIGHTS_PATH = '/home/lengjingcheng/codes/iguane_harmonization/harmonization/iguane_weights.h5'
VAL_DATA_PATH      = '/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/preprocessed/'
SAVE_DIR           = './results/iguane_PSNR_SSIM/'
# ===================================================
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------- 模型 ---------------
gen_univ = Generator()
gen_univ.load_weights(MODEL_WEIGHTS_PATH)
gen_univ.trainable = False

# --------------- IO ---------------
def load_nii(path):
    img = nib.load(path).get_fdata().astype(np.float32)
    h, w, d = img.shape
    # print("图片形状： ",h, w, d) 160 192 160
    # # 中心裁剪
    # start_h = max(0, (h - 160) // 2)
    # start_w = max(0, (w - 192) // 2)
    # start_d = max(0, (d - 160) // 2)
    # img = img[start_h:start_h+160, start_w:start_w+192, start_d:start_d+160]
    # # pad
    # pad_h, pad_w, pad_d = 160-img.shape[0], 192-img.shape[1], 160-img.shape[2]
    # img = np.pad(img, ((0,pad_h),(0,pad_w),(0,pad_d)), 'constant')
    return img

def save_nii(arr, affine, path):
    nib.save(nib.Nifti1Image(arr, affine), path)

# --------------- 3-D 推理 ---------------
def infer_3d(vol):
    """vol: [160,192,160] -> harmonized [160,192,160]"""
    mask = vol > 0
    # vol_norm = vol / 500 - 1 为啥要除以500呢，没必要吧
    vol_norm = vol  - 1
    vol_norm[~mask] = 0
    vol_5d = vol_norm[None, ..., None]           # [1,160,192,160,1]
    harm_5d = gen_univ(vol_5d, training=False)   # 3-D 模型直接出
    harm = harm_5d.numpy().squeeze()             # [160,192,160]
    # print(f'[debug] harm {harm.min():.1f} ~ {harm.max():.1f}, 中位数{np.median(harm[mask])}')
    # harm = (harm + 1) * 500
    harm = harm + 1  # 防止分布拉的过大，所以为啥后处理要乘上五百呢
    # print(f'[debug] harm {harm.min():.1f} ~ {harm.max():.1f}, 中位数{np.median(harm[mask])}')
    harm[~mask] = 0
    # print(f'[debug] src  {vol.min():.1f} ~ {vol.max():.1f}, 中位数{np.median(vol[mask])}')
    # print(f'[debug] harm {harm.min():.1f} ~ {harm.max():.1f}, 中位数{np.median(harm[mask])}')
    return np.maximum(harm, 0)


# [debug] 原始mri  -0.0 ~ 2.2
# [debug] 预处理之后，推理之前data  -1.0 ~ 1.2
# [debug] 推理之后data  -1.7 ~ 1.0
# [debug] 后处理之后data  0.0 ~ 982.9

# [debug] gold  0.0 ~ 1.0
# [debug] harm -125.6 ~ 410.6
# [debug] harm -1.3 ~ 1.0


# --------------- 指标 ---------------
def calc_metrics(gt, pred):
    psnr_list, ssim_list = [], []
    for z in range(gt.shape[0]):
        g, p = gt[z], pred[z]
        mask = (g > 0.01).astype(np.float32)
        if mask.sum() < 100:
            continue
        psnr_list.append(psnr(g, p, data_range=g.max()-g.min()))
        ssim_list.append(ssim(g, p, data_range=g.max()-g.min(),
                              win_size=3, mask=mask))
    return psnr_list, ssim_list

# 生成的和gold相比PSNR总是负的
def simulate_source_3d(gold_vol: np.ndarray,
                       gamma_range=(0.7, 1.2),
                       scale_range=(0.85, 1.15),
                       bias_range=(-30, 30),
                       noise_sigma=10.,
                       clip_range=(0, 5000)) -> np.ndarray:
    """
    3-D 模拟源域（借鉴 BlindHarmony 的 gen_sc + gamma 思想）
    gold_vol : (H, W, D) 参考体积，已归一化到 0-1 或 0-5000
    返回     : 同 shape 的模拟源域
    """
    # 1. gamma 变换（模拟不同扫描仪对比度）
    gamma = np.random.uniform(*gamma_range)
    src = gold_vol ** gamma

    # 2. 随机强度比例（模拟增益差异）
    scale = np.random.uniform(*scale_range)
    src *= scale

    # 3. 随机偏置（模拟基线/偏置场）
    bias = np.random.uniform(*bias_range)
    src += bias

    # 4. 高斯噪声（模拟电子噪声）
    if noise_sigma > 0:
        src += np.random.normal(0, noise_sigma, src.shape)

    # 5. 裁剪到合法范围
    src = np.clip(src, *clip_range)

    return src

def gen_sc_3d(gold_vol: np.ndarray, mode: str = 'gamma07') -> np.ndarray:
    """
    3-D 整卷模拟源域（借鉴原 2-D gen_sc 的 gamma/exp/log + min-max）
    gold_vol : (H, W, D) 已归一化到 0-1 或 0-任意正数
    return   : 同 shape 的模拟源域
    """
    # 0. 保险：先压到 0-1
    # gt = (gold_vol - gold_vol.min()) / (gold_vol.max() - gold_vol.min() + 1e-8)
    v_min, v_max = gold_vol.min(), gold_vol.max()
    mask = gold_vol > 0
    gold_vol[~mask] = 0
    # print(f'[debug]gen_sc_3d src  {gold_vol.min():.1f} ~ {gold_vol.max():.1f}, 中位数{np.median(gold_vol[mask])}')
    gt=gold_vol
    # 1. 非线性强度变换（照搬原逻辑）
    if mode == 'exp':
        img_sc = np.exp(gt)
    elif mode == 'log':
        img_sc = np.log1p(3 * gt)          # log1p = log(1+x) 避免 log(0)
    elif mode.startswith('gamma'):
        gamma = float(mode.replace('gamma', ''))  # 'gamma07' -> 0.7
        img_sc = gt ** gamma
    else:
        raise ValueError(f'Unknown mode: {mode}')

    # 2. 再 min-max 一次（与原函数末尾一致）
    img_sc = v_min + (img_sc - img_sc.min()) / (img_sc.max() - img_sc.min() + 1e-8) * (v_max - v_min)


    # print(f'[debug]gen_sc_3d src  {img_sc.min():.1f} ~ {img_sc.max():.1f}, 中位数{np.median(img_sc[mask])}')

    # img_sc[~mask] = 0
    # med = np.median(img_sc[mask])
    # # print(med)
    # if med == 0:
    #     med = 1e-8
    # img_sc = img_sc / med     
    # print(f'[debug]gen_sc_3d src  {img_sc.min():.1f} ~ {img_sc.max():.1f}, 中位数{np.median(img_sc[mask])}')

    return img_sc

# --------------- 主流程 ---------------
def main():
    gold_files = sorted([f for f in os.listdir(VAL_DATA_PATH)
                         if f.endswith('.nii.gz') and 'sub-' in f])
    assert gold_files, 'No NIfTI found!'

    all_src_psnr, all_src_ssim = [], []
    all_hm_psnr,  all_hm_ssim  = [], []

    for fname in tqdm(gold_files, desc='3-D inference'):
        sub_id = re.findall(r'sub-\d+', fname)[0]
        gold_path = os.path.join(VAL_DATA_PATH, fname)
        gold_vol = load_nii(gold_path)
        # print(f'[debug] gold_vol  {gold_vol.min():.1f} ~ {gold_vol.max():.1f}')
        mask = gold_vol > 0
        # print(f'[debug] gold_vol  {gold_vol.min():.1f} ~ {gold_vol.max():.1f}, 中位数{np.median(gold_vol)}, 中位数{np.median(gold_vol[gold_vol > 0])}')

        # gold_vol = np.clip(gold_vol, 0, None)   # ← 加这一行，把负数消灭在 gamma 变换之前，防止出现nan

        # 模拟源域（可换成真实源域）
        # src_vol = gold_vol * np.random.uniform(0.85, 1.15) + np.random.uniform(-30, 30)
        # src_vol = np.clip(src_vol, 0, 5000)

        # src_vol = simulate_source_3d(gold_vol,
        #                      gamma_range=(0.7, 1.2),
        #                      scale_range=(0.85, 1.15),
        #                      bias_range=(-30, 30),
        #                      noise_sigma=10.,
        #                      clip_range=(0, 5000))


        src_vol = gen_sc_3d(gold_vol, mode='gamma07')
        # print(f'[debug] src_vol  {src_vol.min():.1f} ~ {src_vol.max():.1f}, 中位数{np.median(src_vol)}, 中位数{np.median(src_vol[src_vol > 0])}')

        # 3-D 推理
        harm_vol = infer_3d(src_vol)
        # print(f'[debug] harm_vol  {harm_vol.min():.1f} ~ {harm_vol.max():.1f}, 中位数{np.median(harm_vol)}, 中位数{np.median(harm_vol[harm_vol > 0])}')

        # 指标
        s_psnr, s_ssim = calc_metrics(gold_vol, src_vol)
        h_psnr, h_ssim = calc_metrics(gold_vol, harm_vol)

        # print(f'PSNR  : {s_psnr[0]:.2f}   {h_psnr[0]:.2f} dB')
        # print(f'SSIM  : {s_ssim[0]:.4f}   {h_ssim[0]:.4f}')
        all_src_psnr.extend(s_psnr); all_src_ssim.extend(s_ssim)
        all_hm_psnr.extend(h_psnr);  all_hm_ssim.extend(h_ssim)

        # 可选：保存体积
        aff = nib.load(gold_path).affine
        save_nii(src_vol,  aff, os.path.join(SAVE_DIR, f'{sub_id}_source.nii.gz'))
        save_nii(harm_vol, aff, os.path.join(SAVE_DIR, f'{sub_id}_harmonized.nii.gz'))

    # 汇总
    def mst(arr): return np.mean(arr), np.std(arr)
    s_pm, s_ps = mst(all_src_psnr); s_sm, s_ss = mst(all_src_ssim)
    h_pm, h_ps = mst(all_hm_psnr);   h_sm, h_hs = mst(all_hm_ssim)

    print('\n=== Original vs Gold ===')
    print(f'PSNR  : {s_pm:.2f} ± {s_ps:.2f} dB')
    print(f'SSIM  : {s_sm:.4f} ± {s_ss:.4f}')
    print('\n=== Harmonized vs Gold ===')
    print(f'PSNR  : {h_pm:.2f} ± {h_ps:.2f} dB')
    print(f'SSIM  : {h_sm:.4f} ± {h_hs:.4f}')
    print(f'\nΔ PSNR: {h_pm-s_pm:+.2f} dB   Δ SSIM: {h_sm-s_sm:+.4f}')

    # CSV
    csv_path = os.path.join(SAVE_DIR, 'metrics_history.csv')
    first = not os.path.exists(csv_path)
    row = {'name': os.path.basename(os.path.dirname(MODEL_WEIGHTS_PATH)) + '_' + \
       os.path.basename(MODEL_WEIGHTS_PATH).split('.')[0],
           'mode': 'val-3d',
           'psnr_improvement': h_pm - s_pm,
           'ssim_improvement': h_sm - s_sm,
           'num_slices': len(all_src_psnr),
           'mean_source_psnr': s_pm,
           'mean_source_ssim': s_sm,
           'mean_harmonized_psnr': h_pm,
           'mean_harmonized_ssim': h_sm}
    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=first, index=False)

    # 画图
    plt.rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(all_src_psnr, bins=30, alpha=0.5, label='Original')
    ax1.hist(all_hm_psnr,  bins=30, alpha=0.5, label='Harmonized')
    ax1.axvline(s_pm, ls='--', color='C0'); ax1.axvline(h_pm, ls='--', color='C1')
    ax1.set_xlabel('PSNR (dB)'); ax1.set_ylabel('# Slices'); ax1.legend(); ax1.grid(alpha=.3)

    ax2.hist(all_src_ssim, bins=30, alpha=0.5, label='Original')
    ax2.hist(all_hm_ssim,  bins=30, alpha=0.5, label='Harmonized')
    ax2.axvline(s_sm, ls='--', color='C0'); ax2.axvline(h_sm, ls='--', color='C1')
    ax2.set_xlabel('SSIM'); ax2.set_ylabel('# Slices'); ax2.legend(); ax2.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'psnr_ssim_distribution_3d.png'), dpi=300)
    print(f'分布图已保存 → {SAVE_DIR}/psnr_ssim_distribution_3d.png')

if __name__ == '__main__':
    main()