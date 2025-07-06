# 导入依赖库
from __future__ import print_function
import time
from keras.constraints import non_neg
from keras.regularizers import l2
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Conv2D, Dropout, Flatten
from keras.layers import LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from losses import SAD, normSAD, resampling, vae_loss
# from datetime import datetime
# from scipy.io import loadmat, savemat
import tensorflow as tf
from utility import SparseReLU, SumToOne
from utility import load_HSI, plotEndmembersAndGT, plotAbundancesSimple, plotAbundancesGT
import os
# import random
import pandas as pd
import scipy.io as sio

# 设置使用GPU设备0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 初始化变量
start_time = time.time()
# batchsize若过小，耗时长,训练效率低。
# 假设batchsize=1,每次用一个数据进行训练,如果数据总量很多时(假设有十万条数据),就需要向模型投十万次数据,完整训练完一遍数据需要很长的时问,训练效率很低;
# 原来这里写的batchsize是20，一个epoch要6min
latent_vec = None

# 数据集名称映射
datasetnames = {'urban': 'Urban4',
                'jasper': 'Jasper',
                'samson': 'Samson',
                'synthetic': 'synthetic5',
                'dc': 'DC',
                'apex': 'apex',
                'moni': 'moni30',
                'houston': 'houston',
                'moffett': 'moffett'}
dataset = "dc"
# 加载高光谱数据，x是数据矩阵，E是端元矩阵
# end, abu, r 是空列表，用于存储端元、丰度和重构误差的结果。
hsi = load_HSI("C:/Users/10070/Desktop/HsU/Datasets/" + datasetnames[dataset] + ".mat")
x = hsi.array()
E = hsi.gt
input_dim = hsi.gt.shape[1]
end = []
abu = []
r = []

# 处理丰度数据，将丰度数据重塑为二维数组。
# s_gt 是丰度数据的真实值，num_cols 是丰度数据的列数。
s_gt = hsi.abundance_gt
num_cols = hsi.abundance_gt.shape[0]
s_gt = np.reshape(s_gt, [num_cols*num_cols, -1])

# 设置超参数
# num_runs 是运行次数。
# latent_dim 和 num_endmembers 是潜在空间的维度和端元数量。
# epochs 是训练轮数。
# batchsize 是批量大小。
# l_vca 是正则化参数。
# use_bias 设置为 False，表示不使用偏置。
# activation_set 是激活函数，使用 LeakyLU。
# initializer 是权重初始化器，使用 glorot_normal。
num_runs = 10
latent_dim = num_endmembers = hsi.gt.shape[0]
epochs = 20
batchsize = x.shape[0] // 10
l_vca = 0.03
l_2 = 0
use_bias = False
activation_set = LeakyReLU(0.2)
initializer = tf.initializers.glorot_normal()


# E_reg 是正则化函数，用于计算端元矩阵的稀疏性。
def Mix_reg(weight_matrix):
    E_reg = l_vca * SAD(weight_matrix, E)
    l2_reg = l2(l_2)    
    return l2_reg(weight_matrix) + E_reg

# 创建AUVAE网络
def create_model(input_dim, latent_dim):
    # auv_input 作为输入的高光谱数据，同时也是对抗生成网络（GAN）的输入。        
    auv_input = Input(shape=(input_dim,), name='auv_input')
    
        
    # U-net结构：对图像预处理 
    # 将auv_input整形以适应U-net的输入
    expand = lambda x: tf.reshape(tf.expand_dims(tf.expand_dims(x, 1), 1), (-1, 1, 1, input_dim))
    unet_input = Lambda(expand, name='unet_input')(auv_input)
    
    
    # Fully Convolutional U-Net :不采用采样层和池化层，只用卷积操作，不改变特征形状，只改变通道数
    # U_net 下采样层 Downsampling
    # stage_1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding="same", name='Conv_1')(unet_input)
    conv1 = BatchNormalization()(conv1)
    # stage_2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding="same", name='Conv_2')(conv1)
    conv2 = BatchNormalization()(conv2)
    # stage_3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding="same", name='Conv_3')(conv2)
    conv3 = BatchNormalization()(conv3)    
    # stage_4
    # Bottleneck layer 瓶颈层
    bottleneck = Conv2D(512, (3, 3), activation='relu', padding="same", name='Bottleneck')(conv3)
    bottleneck = Dropout(0.2)(bottleneck)
    
    # U_net 上采样层 Upsampling
    # stage_5
    deconv1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='Deconv_1')(bottleneck)
    merge1 = Concatenate(name='Merge_1')([conv3, deconv1])
    # stage_6
    deconv2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='Deconv_2')(merge1)
    merge2 = Concatenate(name='Merge_2')([conv2, deconv2])
    # stage_7
    deconv3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='Deconv_3')(merge2)
    merge3 = Concatenate(name='Merge_3')([conv1, deconv3])

    # U-Net 的最终输出
    unet_output = Conv2D(input_dim, (1, 1), kernel_regularizer=l2(1e-3), activation='relu', padding='same', name='Unet_Output')(merge3)
    
    
    # 展平以接入VAE
    flatten = Flatten(name='Flatten')(unet_output)


    # VAE结构
    # encoder 编码器采用 4 层 Dense（全连接层），维度依次递减：9631*端元数
    encode = Dense(latent_dim * 9, activation=activation_set, name='Encoder_1')(flatten)
    encode = Dense(latent_dim * 6, activation=activation_set, name='Encoder_2')(encode)
    encode = Dense(latent_dim * 3, activation=activation_set, name='Encoder_3')(encode)
    encode = Dense(latent_dim, use_bias=use_bias, kernel_initializer=None, activation=activation_set, name='Encoder_4')(encode)        
    

    
    # 批归一化处理
    encode = BatchNormalization()(encode)
    # Soft Thresholding, SparseReLU 进行稀疏约束
    encode = SparseReLU(alpha_initializer='zero', alpha_constraint=non_neg(), activity_regularizer=None)(encode)
    # Sum To One (ASC), SumToOne 确保丰度和为 1（ASC 约束）
    encode = SumToOne(axis=0, name='Abundances', activity_regularizer=None)(encode)
    # 封装encoder模型
    encoder = Model(auv_input, encode, name='Encoder')
    encoder.compile(optimizer=Adam(learning_rate=2e-4), loss="mse")
    

    # decoder 解码器包含一个全连接层，将 latent_dim 维的丰度向量解码为 input_dim 维的光谱数据。
    # kernel_constraint=non_neg() 约束端元为非负数（ANC 约束）；E_reg 正则化端元，避免数值不稳定。
    decode = Dense(input_dim, activation='linear', name='Endmembers', use_bias=use_bias, kernel_constraint=non_neg(), kernel_regularizer=Mix_reg, kernel_initializer=initializer)(encode)
    decoder = Model(encode, decode, name='Decoder')
    decoder.compile(optimizer=Adam(learning_rate=2e-4), loss="mse")
    
    # 解码器输出Y_hat估计值
    Y_hat = Dense(input_dim, activation=activation_set, name='Y_hat')(decode)


    # 封装自编码器模型：U-net + VAE部分  
    autoencoder = Model(auv_input, Y_hat, name='VAE')
    autoencoder.compile(optimizer=Adam(learning_rate=2e-4), loss=normSAD)

    
    # 判别器 通过 sigmoid 预测输入是“真实”还是“生成”的丰度分布。
    discriminate = Dense(intermediate_dim2, input_shape=(latent_dim,), activation='relu', name='Discriminator_1')(encode)
    # discriminate = Dense(intermediate_dim2, activation='relu', name='Discriminator_1')(encode)
    discriminate = Dense(intermediate_dim1, activation='relu', name='Discriminator_2')(discriminate)
    discriminate = Dense(1, activation='sigmoid', name='Discriminator_3')(discriminate)
    
    # 封装判别器模型 discriminator: 监督 encoder 生成合理的丰度分布
    discriminator = Model(encode, discriminate, name='Discriminator')
    discriminator.compile(optimizer=Adam(learning_rate=2e-4), loss="binary_crossentropy")
    discriminator.trainable = False
    
    
    # 封装生成器模型 generator: Encoder → Discriminator
    generator = Model(auv_input, discriminator(encoder(auv_input)), name='Generator')
    generator.compile(optimizer=Adam(learning_rate=2e-4), loss="binary_crossentropy")


    # 输出模型结构与参数    
    print("Autoencoder Architecture")
    print(autoencoder.summary())
    print("Discriminator Architecture")
    print(discriminator.summary())
    print("Generator Architecture")
    print(generator.summary())


    # 绘制模型结构图
    plot_model(autoencoder, to_file="autoencoder_graph.png")
    plot_model(discriminator, to_file="discriminator_graph.png")
    plot_model(generator, to_file="generator_graph.png")

    
    return autoencoder, discriminator, generator, encoder, decoder    



# 训练模型 在每个epoch中，分别训练自动编码器、判别器和生成器；计算并打印损失值；返回潜在向量、端元和重构结果。
# Autoencoder 先训练
# Discriminator 训练真假丰度
# Generator 训练欺骗判别器
def train(batch_size, n_epochs):
    autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim = hsi.gt.shape[1], latent_dim = latent_dim)
    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        discriminator_losses = []
        generator_losses = []

        for batch in range(x.shape[0] // batch_size):
            start = int(batch * batch_size)
            end = int(start + batch_size)
            samples = x[start:end]

            autoencoder_history = autoencoder.fit(x=samples, y=samples, epochs=30, batch_size=batch_size, validation_split=0.0, verbose=0)

            fake_latent = encoder.predict(samples)
            real_sample = s_gt[start:end]
            real_sample = np.random.normal(size=(batch_size, latent_dim)) # 真实样本输入判别器替换为了标准正态分布
            discriminator_input = np.concatenate((fake_latent, real_sample))
            discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=10, batch_size=batch_size, validation_split=0.0, verbose=0)
            generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=10, batch_size=batch_size, validation_split=0.0, verbose=0)
            autoencoder_losses.append(autoencoder_history.history["loss"])

            discriminator_losses.append(discriminator_history.history["loss"])
            generator_losses.append(generator_history.history["loss"])
        print("\nEpoch {}/{} ".format(epoch, n_epochs, ))
        print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
        print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
        print("Generator Loss: {}".format(np.mean(generator_losses)))

    z_latent = encoder.predict(x)
    endmember = decoder.get_weights()[0]
    reconstruction = decoder.predict(z_latent)
    return z_latent, endmember, reconstruction


# 主程序 设置中间层的维度；创建输出文件夹；循环运行模型多次，保存结果并绘制图表；计算并保存端元、丰度和重构误差的结果。
if __name__ == "__main__":
    global desc, intermediate_dim1, intermediate_dim2, intermediate_dim3
    original_dim = hsi.gt.shape[1]
    intermediate_dim1 = int(np.ceil(original_dim * 1.2) + 5)
    intermediate_dim2 = int(max(np.ceil(original_dim / 4), latent_dim + 2) + 3)
    intermediate_dim3 = int(max(np.ceil(original_dim / 10), latent_dim + 1))

    desc = "auae"



    output_path = 'C:/Users/10070/Desktop/HsU/Results'
    method_name = 'AUAeNet'
    mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
    endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
    abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
    if not os.path.exists(mat_folder):
        os.makedirs(mat_folder)
    if not os.path.exists(endmember_folder):
        os.makedirs(endmember_folder)
    if not os.path.exists(abundance_folder):
        os.makedirs(abundance_folder)

    for run in range(num_runs):
        # 固定随机种子，确保可重复性
        """
        random_seed = 1
        random.seed(random_seed)  # set random seed for python
        np.random.seed(random_seed)  # set random seed for numpy
        tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
        os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu
        """
        endmember_name = datasetnames[dataset] + '_run' + str(run)
        endmember_path = endmember_folder + '/' + endmember_name
        abundance_name = datasetnames[dataset] + '_run' + str(run)
        abundance_path = abundance_folder + '/' + abundance_name

        print('Start Running! run:', run+1)

        z_latent, endmember, re = train(batch_size=batchsize, n_epochs=epochs)
        z_latent = np.reshape(z_latent, [num_cols, num_cols, -1])
        plotEndmembersAndGT(endmember, hsi.gt, endmember_path, end)
        plotAbundancesSimple(z_latent, hsi.abundance_gt, abundance_path, abu)
        armse_y = np.sqrt(np.mean(np.mean((re - x) ** 2, axis=1)))
        r.append(armse_y)
        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': endmember,
                                                                                  'A': z_latent,
                                                                                  'Y_hat': re
                                                                                  })
        print('-' * 70)

    end = np.reshape(end, (-1, num_endmembers + 1))
    abu = np.reshape(abu, (-1, num_endmembers + 1))
    dt = pd.DataFrame(end)
    dt2 = pd.DataFrame(abu)
    dt3 = pd.DataFrame(r)
    dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
        dataset] + '各端元SAD及mSAD运行结果.csv')
    dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
        dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
    dt3.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
        dataset] + '重构误差RE运行结果.csv')
    abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
        dataset] + '参照丰度图'
    plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
    end_time = time.time()
    print('程序运行时间为：', end_time - start_time, 's')
