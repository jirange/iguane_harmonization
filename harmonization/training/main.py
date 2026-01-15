# Mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow import concat as tf_concat
from json import dump as json_dump


# 放在所有 import 最后即可
import os, datetime
import tensorflow as tf
import nibabel as nib
import tensorflow as tf
from skimage.metrics import structural_similarity
import re
import json
import gc
import tensorflow as tf
#################INPUT DATASETS#########################
from input_pipeline.tf_dataset import datasets_from_tfrecords, datasets_from_tfrecords_biasSampling
# dataset_pairs = # TO DEFINE


# 2. 核心参数配置（关键：IXI是源，SALD是目标）
# 目标域（Reference）：SALD各站点的TFRecord路径（单文件/多文件列表）
# records_ref = ["/home/lengjingcheng/datasets/tfrecords/sald_target.records.gz",]
# 在重新做预处理之后再训练
records_ref = ["/home/lengjingcheng/datasets/tfrecords/SALD_Unkonw_preprocessed.records.gz"]
# records_ref = ["/home/lengjingcheng/datasets/tfrecords/traveler_ref_ses-NOT1ACH001.records.gz",]

# 源域（Source）：IXI数据集的TFRecord路径（列表的列表，每个子列表对应1个源域（此处仅IXI一个源域））
# 注意：records_src是「源域列表」，此处IXI作为唯一源域，故是包含1个子列表的列表

# IXI 数据集，HH[185]是3T，Guys[322]和IOP[74]都是1.5T
# records_src = [
#     ["/home/lengjingcheng/datasets/tfrecords/ixi_source_Guys.records.gz"],  # 源域1：IXI-Guys
#     ["/home/lengjingcheng/datasets/tfrecords/ixi_source_HH.records.gz"],    # 源域2：IXI-HH
#     ["/home/lengjingcheng/datasets/tfrecords/ixi_source_IOP.records.gz"]    # 源域3：IXI-IOP
# ]

# records_src = [
#     ["/home/lengjingcheng/datasets/tfrecords/traveler_src_ses-NOT2ING001.records.gz"],  # 源域1：20
#     ["/home/lengjingcheng/datasets/tfrecords/traveler_src_ses-OXF1PRI001.records.gz"],  # 源域2：20
#     ["/home/lengjingcheng/datasets/tfrecords/traveler_src_ses-OXF2PRI001.records.gz"],  # 源域3：20
# ]

records_src = [
    ["/home/lengjingcheng/datasets/tfrecords/ixi_Guys_preprocessed.records.gz" ],  # 源域1：IXI-Guys 322
    ["/home/lengjingcheng/datasets/tfrecords/traveler_src_ABIDEII-KKI_1.records.gz"],    # 源域2：ABIDE II-KKI_1 211
    ["/home/lengjingcheng/datasets/tfrecords/traveler_ref_ses-1.records.gz"]    # 源域3： ABIDE 后面 600~1112=512
]
# records_src = [
#     "/home/lengjingcheng/datasets/tfrecords/ixi_source_Guys.records.gz",  # 源域1：IXI-Guys
# ]
# Shuffle缓冲区大小（建议为对应数据集样本数的50%）
buf_size_ref = 100  # SALD目标域总样本数≈600 → 缓冲区300（可根据实际样本数调整）
# buf_sizes_src = [150, 100, 50]  # 对应Guys、HH、IOP三家医院的缓冲区
buf_sizes_src=[100,100,100]
# 批次大小（3D MRI显存限制，建议=1）
BATCH_SIZE = 1

# 3. 构建无偏采样的dataset_pairs
dataset_pairs = datasets_from_tfrecords(
    records_ref=records_ref,        # 目标域：SALD
    records_src=records_src,        # 源域：IXI（唯一源域）
    buf_size_ref=buf_size_ref,      # SALD的shuffle缓冲区
    buf_sizes_src=buf_sizes_src,    # IXI的shuffle缓冲区（列表长度=源域数）
    batch_size=BATCH_SIZE,
    compression_type='GZIP'         # 与TFRecord转换时的压缩格式一致（未压缩则设为None）
)

# 4. 验证取数（可选，确认形状匹配）
if len(dataset_pairs) > 0:
    batch_target, batch_source = next(dataset_pairs[0])  # 取唯一源域（IXI）的批次
    print(f"目标域图像（SALD）形状：{batch_target.shape}")  # 预期：(1, 182, 218, 182, 1)
    print(f"源域图像（IXI）形状：{batch_source.shape}")    # 预期：(1, 182, 218, 182, 1)
    print("dataset_pairs构建成功！")
else:
    print("dataset_pairs为空，请检查TFRecord路径是否正确！")


# dataset_pairs = datasets_from_tfrecords(
#     records_ref=records_ref,
#     records_src=records_src,
#     buf_size_ref=500,
#     buf_sizes_src= [300, 200] ,
#     batch_size=1,
#     compression_type='GZIP'  # 与TFRecord生成时的压缩格式一致
# )


##########INPUT PARAMETERS################
DEST_DIR_PATH = "/home/lengjingcheng/codes/iguane_harmonization/harmonization/my_train_SALD-ABIDE-IXI3-check"
# TO DEFINE # 模型/日志保存路径
N_EPOCHS = 100  # 总训练轮数
STEPS_PER_EPOCH = 200  # 每轮训练步数（按批次算）

SAVE_EVERY = 10  # 每 10  epoch 存一次最新权重
LATEST_GEN_PATH = DEST_DIR_PATH + '/latest_genUniv.h5'


CKPT_JSON = os.path.join(DEST_DIR_PATH, 'ckpt.json')

def save_checkpoint(epoch, step_counter, best_score, record_dict):
    meta = {'epoch': epoch, 'step_counter': step_counter,
            'best_score': best_score, 'record_dict': record_dict}
    with open(CKPT_JSON, 'w') as f:
        json_dump(meta, f, indent=2)

def try_load_checkpoint():
    if not os.path.exists(CKPT_JSON):
        return 1, 0, None, {}          # 从头训练
    with open(CKPT_JSON) as f:
        meta = json.load(f)
    epoch0 = meta['epoch'] + 1        # 从下一个 epoch 开始
    step_counter = meta['step_counter']
    best_score = meta['best_score']
    record_dict = meta['record_dict']
    # 恢复最新权重
    gen_univ.load_weights(LATEST_GEN_PATH)
    for i in range(len(dataset_pairs)):
        gens_bwd[i].load_weights(f"{DEST_DIR_PATH}/latest_genBwd_{i+1}.h5")
        discs_bwd[i].load_weights(f"{DEST_DIR_PATH}/latest_discBwd_{i+1}.h5")
        discs_ref[i].load_weights(f"{DEST_DIR_PATH}/latest_discRef_{i+1}.h5")
    print(f'[checkpoint loaded] resume from epoch {epoch0}')
    return epoch0, step_counter, best_score, record_dict

# Instancitation of the generators and the discriminators
import sys
sys.path.append('..')
from model_architectures import Generator, Discriminator
gen_univ = Generator()
gens_bwd = [Generator() for _ in range(len(dataset_pairs))]
discs_ref = [Discriminator() for _ in range(len(dataset_pairs))]
discs_bwd = [Discriminator() for _ in range(len(dataset_pairs))]
# len(dataset_pairs) = 站点数
# 每个站点配 1 个反向生成器 + 1个源判别器+ 1个参考判别器，全局仅 1 个核心（正向）生成器gen_univ

##################VALIDATION############################
# Definition of an evaluation function for the current model. Takes no argument.
import numpy as np 
# def eval_model(): return np.random.uniform()  # 占位符：实际需替换为真实评估逻辑
def eval_model():
    """
    核心验证逻辑：计算旅行受试者/跨站点验证集的SSIM均值（越高越好）
    仅保留论文核心验证指标，忽略次要维度，适配训练中快速验证的需求
    """
    # ========== 1. 加载核心验证集（提前准备，需满足2个条件） ==========
    # 条件1：未参与训练的跨站点图像，标注「受试者ID」（用于匹配同一人跨扫描仪图像）
    # 条件2：已完成预处理（配准、裁剪至160×192×160、颅骨剥离）
    # （注：val_imgs为numpy数组 [N, 160, 192, 160]，val_sub_ids为受试者ID列表 [sid1, sid2, ...]）
    val_imgs, val_sub_ids = load_core_validation_data()  
    print(f"加载了{len(val_imgs)}张图像")
    # ========== 2. 生成协调后图像（固定生成器，避免验证时更新参数） ==========
    gen_univ.trainable = False  # 关键：冻结生成器权重
    harmonized_imgs = []
    for img in val_imgs:
        # 预处理（与训练逻辑完全一致）
        mask = img > 0  # 脑区掩码
        img_norm = img / 500 - 1  # 强度归一化（500为脑区强度中位数，可按论文调整）
        img_norm[~mask] = 0  # 背景置0
        
        # 模型推理（添加batch/channel维度）
        img_input = np.expand_dims(img_norm, axis=(0, 4))  # shape: [1, 160, 192, 160, 1]
        img_tensor = tf.convert_to_tensor(img_input, dtype='float32')
        harmonized = gen_univ(img_tensor, training=False).numpy().squeeze()  # 移除多余维度
        
        # 后处理（恢复原始强度范围）
        harmonized = (harmonized + 1) * 500
        harmonized[~mask] = 0
        harmonized = np.maximum(harmonized, 0)  # 确保非负
        harmonized_imgs.append(harmonized)
    gen_univ.trainable = True  # 解冻生成器

    # ========== 3. 计算核心指标：同一受试者跨站点图像对的SSIM均值 ==========
    ssim_scores = []
    # 遍历每个受试者，匹配其跨站点图像对
    unique_subs = np.unique(val_sub_ids)
    for sub_id in unique_subs:
        # 筛选该受试者的所有协调后图像（跨站点/扫描仪）
        sub_harmonized = [h for h, sid in zip(harmonized_imgs, val_sub_ids) if sid == sub_id]
        # 计算该受试者所有图像对的SSIM
        n_imgs = len(sub_harmonized)
        if n_imgs < 2:
            continue  # 至少2张图像才计算SSIM
        for i in range(n_imgs):
            for j in range(i+1, n_imgs):
                # 论文指定SSIM参数：3D计算、win_size=7、data_range=99百分位数
                max_val = np.percentile(np.concatenate([sub_harmonized[i].ravel(), sub_harmonized[j].ravel()]), 99)
                ssim_val = structural_similarity(
                    sub_harmonized[i], sub_harmonized[j],
                    data_range=max_val,
                    win_size=7,
                    multichannel=False  # 3D单通道图像
                )
                ssim_scores.append(ssim_val)
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    print(f"\n✅ 验证完成：有效图像对{len(ssim_scores)}组 | 平均SSIM={avg_ssim:.4f}")
   # ========== 4. 返回核心评分（SSIM均值，越高代表技术差异消除效果越好） ==========
    return avg_ssim

# 辅助函数：加载核心验证数据（需根据数据集路径自定义实现）
def load_core_validation_data():
    """
    加载预准备的核心验证集：
    - 输出1：val_imgs (numpy数组)，shape=(N, 160, 192, 160)
    - 输出2：val_sub_ids (列表)，长度=N，每个元素为受试者ID（如sub-16981）
    """
    # 示例逻辑（需替换为实际数据集路径）
    val_data_path = "/home/lengjingcheng/codes/iguane_harmonization/data/ON-Harmony/preprocessed/"  # 提前划分的核心验证集路径
    val_imgs = []
    val_sub_ids = []
    # paths = os.listdir(val_data_path)
    paths = sorted(os.listdir(val_data_path))[:20]
    # 遍历验证集文件，加载图像和对应的受试者ID
    for file_name in paths:
        if file_name.endswith(".nii.gz"):
            # 加载NIfTI图像
            img = nib.load(os.path.join(val_data_path, file_name)).get_fdata()
            val_imgs.append(img)
            # 解析受试者ID（如从文件名sub-16981_ses-xxx_T1w.nii.gz中提取）
            sub_id = re.findall(r"sub-\d+", file_name)[0]
            val_sub_ids.append(sub_id)
    return np.array(val_imgs), val_sub_ids

EVAL_FREQ = 5 # evaluates the model every X epochs
#########################################################


# Initialization of the optimizers
# 优化器初始化：Adam + 多项式学习率衰减 + 混合精度损失缩放
INIT_LR = 0.0002
END_LR = 0.00002
#  计算总步数（不同模型训练步数不同）
n_steps_gen_univ = N_EPOCHS*STEPS_PER_EPOCH*len(dataset_pairs)  # 全局生成器总步数（多站点累加）
n_steps_gen_bwd = N_EPOCHS*STEPS_PER_EPOCH
n_steps_discs = N_EPOCHS*STEPS_PER_EPOCH

def optimizer(n_steps):
    # PolynomialDecay 多项式学习率衰减：初始LR线性下降到END_LR，避免后期震荡
    # mixed_precision.LossScaleOptimizer 混合精度损失缩放：解决FP16数值下溢，动态调整损失缩放因子        
    opt = Adam(learning_rate=PolynomialDecay(INIT_LR, n_steps, END_LR))
    # 混合精度训练的核心组件 ——FP16 计算易出现数值下溢（梯度接近 0），通过缩放损失值保证梯度有效，训练后再还原，不影响最终梯度更新；
    return mixed_precision.LossScaleOptimizer(opt, dynamic=True)
# 为不同模型分配优化器
genUnivOptimizer = optimizer(n_steps_gen_univ)
genOptimizers_bwd = [optimizer(n_steps_gen_bwd) for _ in range(len(dataset_pairs))]
discOptimizers_ref = [optimizer(n_steps_discs) for _ in range(len(dataset_pairs))]
discOptimizers_bwd = [optimizer(n_steps_discs) for _ in range(len(dataset_pairs))]


# Instantiation of the training objects
from trainers import Discriminator_trainer, Generator_trainer
# Generator_trainer 自定义类，封装生成器的损失计算（对抗损失 + 循环损失 + 身份损失）和梯度反向传播
# 输入：全局生成器、该站点反向生成器、两个判别器、对应优化器；
# Discriminator_trainer 自定义类，封装判别器的对抗损失计算和梯度更新；
# 输入：判别器、对应的生成器、优化器；
genTrainers = [Generator_trainer(gen_univ, gens_bwd[i], discs_ref[i], discs_bwd[i], genUnivOptimizer, genOptimizers_bwd[i]) for i in range(len(dataset_pairs))]
discTrainers_ref = [Discriminator_trainer(discs_ref[i], gen_univ, discOptimizers_ref[i]) for i in range(len(dataset_pairs))]
discTrainers_src = [Discriminator_trainer(discs_bwd[i], gens_bwd[i], discOptimizers_bwd[i]) for i in range(len(dataset_pairs))]


# Main training function
DISC_N_BATCHS = 2 # number of batches to train the discriminators
# 判别器每次训练的批次数量（先训练判别器，稳定GAN）
indices_sites = np.arange(len(dataset_pairs))
# def train_step(step_counter):
#     np.random.shuffle(indices_sites)  # 打乱站点顺序，避免训练偏向
#     results = {'genFwd_idLoss':0}
#     for idSite in indices_sites:
#         # 加载判别器训练的批次（DISC_N_BATCHS*2批：一半训判别器，一半留作生成器训练）
#         batchs = tf_concat([dataset_pairs[idSite].get_next() for _ in range(DISC_N_BATCHS*2)], axis=1)
#         imagesRef = batchs[0]
#         imagesSrc = batchs[1]

#         # 1. 训练判别器（先训判别器，GAN训练的稳定技巧）
#         results[f"discRef_{idSite+1}_loss"] = discTrainers_ref[idSite].train(imagesRef[DISC_N_BATCHS:], imagesSrc[DISC_N_BATCHS:])
#         results[f"discSrc_{idSite+1}_loss"] = discTrainers_src[idSite].train(imagesSrc[:DISC_N_BATCHS], imagesRef[:DISC_N_BATCHS])
#         # 2. 训练生成器（用新批次，避免判别器数据泄露）
#         batchRef, batchSrc = dataset_pairs[idSite].get_next()

#         (results[f'genFwd_adv_loss_{idSite+1}'], results[f'genBwd_adv_loss_{idSite+1}'], results[f'cycle_loss_refSref_{idSite+1}'],
#          results[f'cycle_loss_srcRsrc_{idSite+1}'], genFwd_idLoss, results[f'genBwd_idLoss_{idSite+1}']) = genTrainers[idSite].train(batchSrc, batchRef)
#         results['genFwd_idLoss'] += (genFwd_idLoss/len(dataset_pairs))
#     #  ======== 新增TensorBoard：每 step 写一次 ========
#     with train_summary_writer.as_default():
#         for k, v in results.items():
#             tf.summary.scalar(k, v.numpy(), step=step_counter)

#     return results

# --- 1. 定义核心静态图训练函数 ---
@tf.function
def site_train_step_graph(img_ref_disc, img_src_disc, img_ref_gen, img_src_gen, 
                          d_trainer_ref, d_trainer_src, g_trainer):
    """
    这是显存回收的关键！所有的梯度计算都在这个静态图中完成。
    只接收数据和具体的 trainer 实例
    """
    print(">>> 警告：正在编译（Tracing）计算图，如果每个 Step 都看到这行，说明内存泄漏就在这！")
    # 直接调用传入的 trainer 实例
    # 训练判别器
    # 注意：这里假设你的 trainer.train 是用 tf.GradientTape 写的
    loss_disc_ref = d_trainer_ref.train(img_ref_disc, img_src_disc)
    loss_disc_src = d_trainer_src.train(img_src_disc, img_ref_disc)
    # 训练生成器
    losses_gen = g_trainer.train(img_src_gen, img_ref_gen)
    # 返回所有损失，TF 会将其包装成张量
    
    return loss_disc_ref, loss_disc_src, losses_gen

# --- 2. 修改调度函数 (Python 逻辑层) ---
def train_step(step_counter):
    np.random.shuffle(indices_sites)
    results = {'genFwd_idLoss': 0.0}
    
    for idSite in indices_sites:
        # 获取数据 (保持不变)
        # batchs = tf_concat([dataset_pairs[idSite].get_next() for _ in range(DISC_N_BATCHS*2)], axis=1)
        # img_ref_disc, img_src_disc = batchs[0][DISC_N_BATCHS:], batchs[1][DISC_N_BATCHS:]
        # batch_ref_gen, batch_src_gen = dataset_pairs[idSite].get_next()

        # # --- 关键修改：在 Python 侧通过 idSite 取出 trainer 对象 --- for TypeError: list indices must be integers or slices, not Tensor
        # current_d_ref_trainer = discTrainers_ref[idSite]
        # current_d_src_trainer = discTrainers_src[idSite]
        # current_g_trainer = genTrainers[idSite]

        # # 将具体的 trainer 对象传入静态图
        # d_ref_l, d_src_l, g_ls = site_train_step_graph(
        #     img_ref_disc, img_src_disc, batch_ref_gen, batch_src_gen,
        #     current_d_ref_trainer, current_d_src_trainer, current_g_trainer
        # )

        # 直接获取 Tensor，不要合并，减少内存副本
        # 判别器需要的 2 组数据
        d_ref_1 = dataset_pairs[idSite].get_next()[0]
        d_src_1 = dataset_pairs[idSite].get_next()[1]
        
        # 生成器需要的 1 组数据
        g_ref, g_src = dataset_pairs[idSite].get_next()

        # 传入静态图
        d_ref_l, d_src_l, g_ls = site_train_step_graph(
            d_ref_1, d_src_1, g_ref, g_src,
            discTrainers_ref[idSite], discTrainers_src[idSite], genTrainers[idSite]
        )

        # C. 核心修复：立即将 Tensor 转为标量 (float)
        # 这样 results 字典里存的是 Python 数字，而不是持有计算图引用的 Tensor 对象
        results[f"discRef_{idSite+1}_loss"] = float(d_ref_l)
        results[f"discSrc_{idSite+1}_loss"] = float(d_src_l)
        
        # 拆解生成器损失 (假设返回的是列表/元组)
        results[f'genFwd_adv_loss_{idSite+1}'] = float(g_ls[0])
        results[f'genBwd_adv_loss_{idSite+1}'] = float(g_ls[1])
        results[f'cycle_loss_refSref_{idSite+1}'] = float(g_ls[2])
        results[f'cycle_loss_srcRsrc_{idSite+1}'] = float(g_ls[3])
        
        id_loss_fwd = float(g_ls[4])
        results[f'genBwd_idLoss_{idSite+1}'] = float(g_ls[5])
        
        results['genFwd_idLoss'] += (id_loss_fwd / len(dataset_pairs))

        del d_ref_1, d_src_1, g_ref, g_src, d_ref_l, d_src_l, g_ls
        gc.collect() # 强制清理

    # D. 写入 TensorBoard (已经在外部处理，这里只需确保值是数字)
    with train_summary_writer.as_default():
        for k, v in results.items():
            tf.summary.scalar(k, v, step=step_counter)
    # 每个 Step 结束后手动清理垃圾
    gc.collect()    

    return results

# ======== TensorBoard：创建 summary writer  ========
log_dir = os.path.join(DEST_DIR_PATH, "logs",
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
train_summary_writer = tf.summary.create_file_writer(log_dir)
print('TensorBoard log dir:', log_dir)
step_counter = 0          # 全局步数

# Training execution
BEST_GEN_PATH = DEST_DIR_PATH+'/best_genUniv.h5'
# 原来这三行
# record_dict = {} # 记录每轮的平均损失
# best_score = None # 最优验证分数（保存最优模型）
# for epoch in range(1,N_EPOCHS+1):
import ctypes
# 改成下面 4 行
start_epoch, step_counter, best_score, record_dict = try_load_checkpoint()
for epoch in range(start_epoch, N_EPOCHS+1):

    tmp_record = {}  # 临时记录当前轮的损失累加
    for step in range(1,STEPS_PER_EPOCH+1):
        res = train_step(step_counter)
        step_counter += 1 # 新加逻辑，为了tensorboard

        if not tmp_record:
            # 修复点：直接赋值，不要调用 .numpy() AttributeError: 'float' object has no attribute 'numpy'
            for key in res.keys(): tmp_record[key] = res[key]
        else:
            for key in res.keys(): tmp_record[key] += res[key]
        
        log = f"End step {step}/{STEPS_PER_EPOCH} époque {epoch}/{N_EPOCHS} | "
        for k in sorted(res.keys()): log += f"{k} = {res[k]:.4f},  "
        print(log, end=f"{' '*20}\r")
        
    if not record_dict:
        for key in sorted(tmp_record.keys()):
            record_dict[key] = [tmp_record[key]/STEPS_PER_EPOCH]
    else:
        for key in tmp_record:
            record_dict[key].append(tmp_record[key]/STEPS_PER_EPOCH)
            
    # ======== tensorboard 新增：写 epoch 平均 ========
    epoch_mean = {k: (tmp_record[k]/STEPS_PER_EPOCH) for k in tmp_record}
    with train_summary_writer.as_default():
        for k, v in epoch_mean.items():
            tf.summary.scalar(k+'_epoch', v, step=epoch)


    log = f"Fin époque {epoch} -> "
    for key,value in record_dict.items():
        log += f"{key} : {value[-1]:.4f}, "
    print(log+' '*20)
    print()

    save_checkpoint(epoch, step_counter, best_score, record_dict)   # 新增
    # ========== 新增：定时保存最新权重 ==========
    if epoch % SAVE_EVERY == 0:
        # 1. 先清理内存，释放临时张量
        tf.keras.backend.clear_session()
        gc.collect()
        gen_univ.save_weights(LATEST_GEN_PATH)
        for i in range(len(dataset_pairs)):
            gens_bwd[i].save_weights(f"{DEST_DIR_PATH}/latest_genBwd_{i+1}.h5")
            discs_bwd[i].save_weights(f"{DEST_DIR_PATH}/latest_discBwd_{i+1}.h5")
            discs_ref[i].save_weights(f"{DEST_DIR_PATH}/latest_discRef_{i+1}.h5")
            # 每保存一组就清理一次内存
            gc.collect()
            tf.keras.backend.clear_session()
        print(f'[latest weights saved at epoch {epoch}]')

    if epoch % EVAL_FREQ == 0:
        score = eval_model()
        print(f"Validation function, score = {score:.3f}")
        if not best_score or score>best_score:
            best_score=score
            gen_univ.save_weights(BEST_GEN_PATH)
            print('New best model saved')
    # 每个 Epoch 结束时调用    
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0) # 强制释放系统内存
    except:
        pass            
            
# Saving of all the models
gen_univ.save_weights(DEST_DIR_PATH+'/generator_univ.h5')
for i in range(len(dataset_pairs)):
    gens_bwd[i].save_weights(f"{DEST_DIR_PATH}/genBwd_{i+1}.h5")
    discs_bwd[i].save_weights(f"{DEST_DIR_PATH}/discBwd_{i+1}.h5")
    discs_ref[i].save_weights(f"{DEST_DIR_PATH}/discRef_{i+1}.h5")
with open(DEST_DIR_PATH+'/stats.json','w') as f: json_dump(record_dict, f)
