import sys
import getopt
import pandas as pd
from time import time
import pipeline

# import cv2
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim


from tqdm import tqdm
# 打印帮助信息
def usage():
    cmds = [
        'python all_in_one.py --in-mri <in_mri> --out_mri <out_mri> [options]',
        'python all_in_one.py --in-csv <in-csv> [options]',
        'python all_in_one.py --just-preprocess --in-mri <in_mri> --out_mri <out_mri> [options]',
    ]
    options = [
        ['--in-mri <input>','input MR image path, requires --out-mri'],
        ['--out-mri <output>','output MR image path, requires --in-mri'],
        ['--in-csv <csv>', 'CSV filepath with two columns: in_paths, out_paths. --in-mri must not be defined'],
        ['--hd-bet-cpu', 'runs HD-BET with cpu (GPU by default)'],
        ['--just-preprocess', 'just-preprocess or inference'],
        ['--n-procs <n>', 'number of CPUs to use if several inputs (--in-csv option), default=1'],
        ['--help', 'displays this help']

    ]
    print('\n'.join(cmds))
    print('Available options are:')
    for opt,doc in options: print(f"\t{opt}{(20-len(opt))*' '}{doc}")

# 计算SSIM的函数（简洁版）
# def calculate_ssim(img1_path, img2_path):
#     """
#     计算两张图像的SSIM值
#     :param img1_path: 原始图像路径
#     :param img2_path: 处理后图像路径
#     :return: SSIM值
#     """
#     # 读取图像（以灰度模式读取，适配MRI图像）
#     img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
#     # 确保两张图像尺寸一致（处理可能的尺寸差异）
#     if img1.shape != img2.shape:
#         img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
#     # 计算SSIM（multichannel=False表示单通道灰度图）
#     ssim_score = ssim(img1, img2, data_range=img2.max() - img2.min())
#     return ssim_score

def main():
    try:
        opts,_ = getopt.getopt(sys.argv[1:], 'h', ['help', 'in-mri=', 'out-mri=', 'in-csv=', 'n-procs=', 'hd-bet-cpu', 'just-preprocess'])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    
    in_mri = None
    out_mri = None
    in_csv = None
    n_procs = 1
    hd_bet_cpu = False
    just_preprocess = False
    for opt,arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        if opt == '--in-mri': in_mri = arg
        elif opt == '--out-mri': out_mri = arg
        elif opt == '--in-csv': in_csv = arg
        elif opt == '--hd-bet-cpu': hd_bet_cpu = True
        elif opt == '--just-preprocess': just_preprocess = True
        elif opt == '--n-procs': n_procs = int(arg)

    if (in_mri is None) ^ (out_mri is None) : 
        usage()
        sys.exit(2)
        
    if not ((in_mri is None) ^ (in_csv is None)):
        usage()
        sys.exit(2)
        
    if in_csv:
        df = pd.read_csv(in_csv)
        # if not just_preprocess:
        #     for _, row in tqdm(df.iterrows(), total=len(df), desc="preprocessing+inferencing"):
        #         pipeline.run_singleproc(row['in_mri'], row['out_mri'], hd_bet_cpu)
        # else:
        #     for _, row in tqdm(df.iterrows(), total=len(df), desc="just preprocessing"):
        #         pipeline.run_singleproc(row['in_mri'], row['out_mri'], hd_bet_cpu, just_preprocess=True)            
         # 批量处理：读取CSV，调用多进程处理 
         # 上面的是伪并行，下面才是真的
        # pipeline.run_multiproc(df.in_paths.tolist(), df.out_paths.tolist(), n_procs, hd_bet_cpu)
        # last_500_in = df['in_mri'].tail(10).tolist()
        # last_500_out = df['out_mri'].tail(10).tolist()
        # pipeline.run_multiproc(last_500_in, last_500_out, n_procs, hd_bet_cpu)

        pipeline.run_multiproc(df.in_mri.tolist(), df.out_mri.tolist(), n_procs, hd_bet_cpu)

    else: # 单文件处理：调用单进程处理
        pipeline.run_singleproc(in_mri, out_mri, hd_bet_cpu,just_preprocess=just_preprocess)
        # else:
        # # 新增：计算并输出SSIM
        #     try:
        #         ssim_score = calculate_ssim(in_mri, out_mri)
        #         print(f"\nSSIM (结构相似性指数) between input and output: {ssim_score:.4f}")
        #         print("SSIM值越接近1，表示两张图像结构相似度越高")
        #     except Exception as e:
        #         print(f"\n计算SSIM时出错: {e}")
        #         print("请检查输入输出图像路径是否正确，或图像文件是否损坏")
    
if __name__ == "__main__":
    t_start = time()
    main()
    print(f"End of execution, total processing time = {time()-t_start:.0f} seconds")

