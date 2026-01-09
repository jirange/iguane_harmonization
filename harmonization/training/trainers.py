from tensorflow import function as tf_function, where as tf_where, GradientTape
from tensorflow.math import reduce_mean, square, abs as tf_abs

LAMBDA_CYC = 30 # 循环损失的权重超参数（CycleGAN核心）
# 循环损失权重LAMBDA_CYC=30：远大于对抗损失，优先保证解剖结构不变（医学图像中结构正确性＞风格匹配）；

# 判别器训练：先训判别器→生成假样本→计算 LSGAN 损失→更新判别器参数；
# 生成器训练：后训生成器→计算对抗 / 循环 / 身份损失→加权求和→更新正向 / 反向生成器参数；

class Discriminator_trainer:
  
    def __init__(self, discriminator, generator, optimizer):
        self.discriminator = discriminator# 待训练的判别器
        self.generator = generator# 对应生成器
        self.optimizer = optimizer
    
    @tf_function(jit_compile=True)  # 编译为计算图，训练速度提升5~10倍
    def train(self, images1, images2):
        # 步骤1：掩码处理（仅保留脑区，非脑区置0，避免背景噪声干扰损失）
        mask = images2>-1
        images1 = tf_where(images1>-1, images1, 0)
        images2 = tf_where(mask, images2, 0)

        # 步骤2：生成假样本（生成器仅推理，不训练）
        fake_images = tf_where(mask, self.generator(images2, training=False), 0)
        # 步骤3：梯度带记录判别器运算，计算损失+梯度
        with GradientTape() as tape:
            disc_real = self.discriminator(images1, training=True)
            disc_fakes = self.discriminator(fake_images, training=True)
            # LSGAN损失（平方损失，比交叉熵更稳定）：
            # 真实图像：判别器输出应接近1 → (disc_real-1)²
            # 假图像：判别器输出应接近0 → (disc_fakes)²
            disc_loss = reduce_mean(square(disc_real-1)) + reduce_mean(square(disc_fakes))
            # 传统 GAN 用交叉熵损失易出现梯度消失 / 模式崩溃，LSGAN 用平方损失让判别器的梯度更平滑，训练更稳定，尤其适合小批量的医学图像数据。
            # 混合精度：缩放损失（解决FP16数值下溢）
            disc_loss_scaled = self.optimizer.get_scaled_loss(disc_loss)
        # 步骤4：计算梯度并反缩放（恢复真实梯度值）
        grads = tape.gradient(disc_loss_scaled, self.discriminator.trainable_weights)
        grads = self.optimizer.get_unscaled_gradients(grads)  # 反缩放梯度
         # 步骤5：应用梯度更新判别器参数
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        return disc_loss
    # 混合精度梯度流程：
    #     损失缩放：disc_loss → disc_loss_scaled（放大损失，避免FP16下溢）
    #     梯度计算：基于scaled损失算梯度（grads是缩放后的梯度）
    #     梯度反缩放：grads → 原始梯度（get_unscaled_gradients）
    #     梯度更新：用原始梯度更新参数

# 生成器的核心目标是欺骗判别器 + 保证循环重构的解剖结构不变 + 强化风格匹配，包含 CycleGAN 的三大核心损失（对抗损失、循环损失、身份损失），同时训练正向生成器（gen_univ，源→参考）和反向生成器（gen_bwd，参考→源）。    
class Generator_trainer:
  
    def __init__(self, genFwd, genBwd, discFwd, discBwd, optimizerFwd, optimizerBwd):
        self.genFwd = genFwd
        self.genBwd = genBwd
        self.discFwd = discFwd
        self.discBwd = discBwd
        self.optimizerFwd = optimizerFwd
        self.optimizerBwd = optimizerBwd
        
    @tf_function(jit_compile=True)
    def train(self, batchSrc, batchRef):
        maskSrc = batchSrc>-1
        maskRef = batchRef>-1
        batchSrc = tf_where(maskSrc, batchSrc, 0)
        batchRef = tf_where(maskRef, batchRef, 0)
        # persistent=True：默认GradientTape只能调用一次gradient方法，而这里需要为genFwd和genBwd分别计算梯度，故设置persistent=True（允许多次调用），函数结束后 TensorFlow 会自动回收资源。
        with GradientTape(persistent=True) as tape:
            # 真源通过正向生成器生成假参考，真参考经过反向生成器生成假源
            fakesRef = tf_where(maskSrc, self.genFwd(batchSrc, training=True), 0)
            fakesSrc = tf_where(maskRef, self.genBwd(batchRef, training=True), 0)
            
            # adversarial losses 对抗损失
            # 对于正向生成器，我们希望正向判别器认为假参考是真参考，也就是假参考生成接近于1：square(disc_fakesRef-1)
            disc_fakesRef = self.discFwd(fakesRef, training=False)
            # 对于反向生成器，我们希望反向判别器认为假源是真源，也就是假源生成接近于1
            disc_fakesSrc = self.discBwd(fakesSrc, training=False)
            genFwd_adv_loss = reduce_mean(square(disc_fakesRef-1))
            genBwd_adv_loss = reduce_mean(square(disc_fakesSrc-1))
            
            # reconstruction losses 循环损失
            # 假源（真参考经过反向生成器）再经过正向生成器生成的（假假参考cycledRef）应该是真参考batchRef  tf_abs(batchRef-cycledRef)
            # 同理：真源经正向生成器再经反向生成器生成的cycledSrc应该与真源batchSrc同
            cycledRef = tf_where(maskRef, self.genFwd(fakesSrc, training=True), 0)
            cycledSrc = tf_where(maskSrc, self.genBwd(fakesRef, training=True), 0)
            cycle_loss_refSref = reduce_mean(tf_abs(batchRef-cycledRef))
            cycle_loss_srcRsrc = reduce_mean(tf_abs(batchSrc-cycledSrc))
            sum_cycle_loss = cycle_loss_refSref + cycle_loss_srcRsrc
            
            # identity losses for the generators 身份损失
            # 真参考经过正向生成器生成的应该参考风格的真参考idRef应该就是真参考batchRef，源同理
            idRef = tf_where(maskRef, self.genFwd(batchRef, training=True), 0)
            idSrc = tf_where(maskSrc, self.genBwd(batchSrc, training=True), 0)
            genFwd_idLoss = reduce_mean(tf_abs(batchRef-idRef))
            genBwd_idLoss = reduce_mean(tf_abs(batchSrc-idSrc))
            
            # final losses
            # 正向总损失 = 正向对抗损失 + 30*循环损失（正+反） + 15*正向身份损失
            genFwd_loss = genFwd_adv_loss + LAMBDA_CYC*sum_cycle_loss + 0.5*LAMBDA_CYC*genFwd_idLoss
            genFwd_loss = self.optimizerFwd.get_scaled_loss(genFwd_loss) # genUnivOptimizer var globale
            genBwd_loss = genBwd_adv_loss + LAMBDA_CYC*sum_cycle_loss + 0.5*LAMBDA_CYC*genBwd_idLoss
            genBwd_loss = self.optimizerBwd.get_scaled_loss(genBwd_loss)
            
        grads_genFwd = tape.gradient(genFwd_loss, self.genFwd.trainable_weights)
        grads_genFwd = self.optimizerFwd.get_unscaled_gradients(grads_genFwd)
        self.optimizerFwd.apply_gradients(zip(grads_genFwd, self.genFwd.trainable_weights))

        grads_genBwd = tape.gradient(genBwd_loss, self.genBwd.trainable_weights)
        grads_genBwd = self.optimizerBwd.get_unscaled_gradients(grads_genBwd)
        self.optimizerBwd.apply_gradients(zip(grads_genBwd, self.genBwd.trainable_weights))

        return genFwd_adv_loss, genBwd_adv_loss, cycle_loss_refSref, cycle_loss_srcRsrc, genFwd_idLoss, genBwd_idLoss