from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Layer, Conv3D, Conv3DTranspose, ReLU, Activation, Add, LeakyReLU, Concatenate, Lambda
from tensorflow.math import reduce_mean, abs as tf_abs
from tensorflow import pad as tf_pad

# 自定义 InstanceNormalization 层、Generator（生成器）和 Discriminator（判别器）
# 这是一个3D U-Net 风格的残差生成器 + PatchGAN 判别器的 GAN 架构（类似 CycleGAN/DeepHarmony），专为 3D MRI 图像同质化设计，核心目标是消除跨设备 / 中心的 MRI 图像差异



# 作用：实例归一化，GAN 中生成器常用，相比 BatchNorm 更适合单样本的风格迁移 / 同质化（避免批次依赖），且适配 3D 数据。
# 初始化：epsilon 防止除零，dtype='float32' 保证精度。
# build 方法：创建可训练的 scale（缩放）和 offset（偏移）参数，shape 是通道数（最后一维），axis 是计算均值的维度（3D 数据的 H/W/D，即 axis=1,2,3）。
# call 方法：计算步骤（均值→偏差→平均绝对偏差→归一化→缩放偏移），注意这里用的是 mean_abs_dev（平均绝对偏差）而非方差，是 DeepHarmony 的设计特点，更鲁棒。

# 为什么用实例归一化（InstanceNorm）？

#     GAN 生成器中，InstanceNorm 比 BatchNorm 更适合单样本 / 小批量场景（医学图像批量通常小），避免批次统计依赖；
#     适配医学图像同质化：仅对 “单个样本内部” 做归一化，保留跨样本的解剖结构，仅消除风格 / 强度差异。


# 标准 InstanceNorm 用 “方差” 归一化，这里用 “平均绝对偏差（MAE）”，是 DeepHarmony 的改进，对 MRI 的噪声 / 极端值更鲁棒。

class InstanceNormalization(Layer):
    
    def __init__(self, epsilon=1e-3):
        super().__init__(dtype='float32')
        self.epsilon = epsilon
        
    def build(self, batch_input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=batch_input_shape[-1:],
            initializer="ones",
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=batch_input_shape[-1:],
            initializer="zeros",
            trainable=True
        )
        self.axis = range(1, len(batch_input_shape)-1)
        super().build(batch_input_shape)
        
    def call(self, x):
        mean = reduce_mean(x, axis=self.axis, keepdims=True)
        dev = x-mean
        mean_abs_dev = reduce_mean(tf_abs(dev), axis=self.axis, keepdims=True)
        normalized = dev / (mean_abs_dev+self.epsilon)
        return self.scale*normalized + self.offset

# 生成器: 采用编码器 - 解码器（U-Net）+ 残差连接
def Generator(image_shape=(None,None,None,1), kernel_initializer='glorot_uniform'):
    # le générateur façon deepHarmony, version résiduelle
    def Conv_block(n_filters):
        return Sequential([
          Conv3D(filters=n_filters, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer),
          ReLU(),
          InstanceNormalization()
        ])

    def Downsample_block(n_filters):
        return Sequential([
          Conv3D(filters=n_filters, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
          ReLU(),
          InstanceNormalization()
        ])

    def Upsample_block(n_filters):
        return Sequential([
          Conv3DTranspose(filters=n_filters, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
          ReLU(),
          InstanceNormalization()
        ])

    inputs = Input(shape=image_shape)
    conv1 = Conv_block(16)(inputs)
    x = Downsample_block(16)(conv1)
    conv2 = Conv_block(32)(x)
    x = Downsample_block(32)(conv2)
    conv3 = Conv_block(64)(x)
    x = Downsample_block(64)(conv3)
    conv4 = Conv_block(128)(x)
    x = Downsample_block(128)(conv4)

    x = Conv_block(256)(x)
    x = Upsample_block(128)(x)
    x = Concatenate()([x,conv4])
    x = Conv_block(128)(x)
    x = Upsample_block(64)(x)
    x = Concatenate()([x,conv3])
    x = Conv_block(64)(x)
    x = Upsample_block(32)(x)
    x = Concatenate()([x,conv2])
    x = Conv_block(32)(x)
    x = Upsample_block(16)(x)
    x = Concatenate()([x,conv1])
    x = Conv_block(16)(x)
    x = Concatenate()([x,inputs])
    last = Sequential([
        Conv3D(filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer),
        Activation('tanh')
    ])(x)
    return Model(inputs=inputs, outputs=Add(dtype='float32')([last,inputs]))

# 判别器：3D PatchGAN 架构
# 判别器采用PatchGAN设计（判断局部 patch 真假，而非整张图），适合医学图像的精细同质化：
def Discriminator(image_shape=(None,None,None,1), lrelu_slope=0.2, kernel_initializer='glorot_uniform'):
    
    def Downsample(n_filters, padding='same', inst_norm=True):
        model = Sequential([
            Conv3D(filters=n_filters, kernel_size=4, strides=2, padding=padding, use_bias=False, kernel_initializer=kernel_initializer)
        ])
        if inst_norm:
            model.add(InstanceNormalization())
        model.add(LeakyReLU(lrelu_slope))
        return model
    
    return Sequential([
        Lambda(lambda x: tf_pad(x, ((0,0),(1,1),(1,1),(1,1),(0,0)), 'CONSTANT', constant_values=-1), input_shape=image_shape),
        Downsample(64, padding='valid', inst_norm=False),
        Downsample(128),
        Downsample(256),
        Conv3D(filters=1, kernel_size=3, strides=1, padding="same", use_bias=True, kernel_initializer=kernel_initializer),
        Activation('linear', dtype='float32')
    ])


