# 导入依赖库
from __future__ import print_function
import time
from keras.constraints import non_neg
from keras.regularizers import l1
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Lambda
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
use_bias = False
activation_set = LeakyReLU(0.2)
initializer = tf.initializers.glorot_normal()


# E_reg 是正则化函数，用于计算端元矩阵的稀疏性。
def E_reg(weight_matrix):
    return l_vca * SAD(weight_matrix, E)

# 创建AVAE网络
def create_model(input_dim, latent_dim):
    # autoencoder_input 作为输入的高光谱数据。
    # generator_input 作为对抗生成网络（GAN）的输入。
    autoencoder_input = Input(shape=(input_dim,))
    generator_input = Input(shape=(input_dim,))

        
    # 编码器部分
    encode = Dense(latent_dim * 9, activation=activation_set, name='Encoder_1')(autoencoder_input)
    encode = Dense(latent_dim * 6, activation=activation_set, name='Encoder_2')(encode)
    encode = Dense(latent_dim * 3, activation=activation_set, name='Encoder_3')(encode)
    encode = Dense(latent_dim, use_bias=use_bias, activation=activation_set, name='Encoder_4')(encode)
    
    # 输出均值 mu 和方差 var
    mu = Dense(latent_dim, use_bias=use_bias, activation=activation_set, name='mu')(encode)
    var = Dense(latent_dim, use_bias=use_bias, activation=activation_set, name='var')(encode)
    # z 是从 mu 和 var 采样而来的潜在向量
    z = Lambda(resampling, output_shape=(latent_dim,), name='z')([mu, var])

    # 在 mu 上添加稀疏 + SumToOne 约束，使其成为最终丰度向量
    mu = BatchNormalization()(mu)
    # z = BatchNormalization()(z)
    mu = SparseReLU(alpha_initializer='zero', alpha_constraint=non_neg(), activity_regularizer=None, name='Sparse_mu')(mu)
    mu = SumToOne(axis=0, name='Abundances', activity_regularizer=None)(mu)  # 最终丰度输出
    
    # 封装 encoder（丰度即 mu）
    encoder = Model(autoencoder_input, mu, name='Encoder')
    encoder.compile(optimizer=Adam(learning_rate=2e-4), loss="mse")

    # 解码器：从 z → 重建光谱
    decode = Dense(input_dim, activation='linear', name='Endmembers', use_bias=use_bias, kernel_constraint=non_neg(), kernel_regularizer=E_reg, kernel_initializer=initializer)(z)
    decoder = Model(z, decode, name='Decoder')
    decoder.compile(optimizer=Adam(learning_rate=2e-4), loss="mse")
    
    # 解码器输出Y_hat估计值
    Y_hat = Dense(input_dim, activation=activation_set, name='Y_hat')(decode)

    # 计算VAE损失函数 vaeloss
    vaeloss = Lambda(vae_loss, output_shape=(1,), name='vae')([autoencoder_input, Y_hat, mu, var])


    # 封装变分自编码器模型（VAE） autoencoder: Encoder → Decoder  
    autoencoder = Model(autoencoder_input, [Y_hat,vaeloss,mu], name='VAE')
    autoencoder.compile(optimizer=Adam(learning_rate=2e-4), loss=[normSAD, "mae", None], loss_weights=[1,1,0]) #原来是1：1 

        
    # 判别器 通过 sigmoid 预测输入是“真实”还是“生成”的丰度分布。
    # discriminate = Dense(intermediate_dim2, input_shape=(latent_dim,), activation='relu', name='Discriminator_1')(mu)
    discriminate = Dense(intermediate_dim2, activation='relu', name='Discriminator_1')(mu)
    discriminate = Dense(intermediate_dim1, activation='relu', name='Discriminator_2')(discriminate)
    discriminate = Dense(1, activation='sigmoid', name='Discriminator_3')(discriminate)
    
    # 封装判别器模型 discriminator: 监督 encoder 生成合理的丰度分布
    discriminator = Model(mu, discriminate, name='Discriminator')
    discriminator.compile(optimizer=Adam(learning_rate=2e-4), loss="binary_crossentropy")
    discriminator.trainable = False
    
    
    # 封装生成器模型 generator: Encoder → Discriminator
    generator = Model(generator_input, discriminator(encoder(generator_input)), name='Generator')
    # generator = Model(generator_input, discriminator(mu), name='Generator')
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
        # 每一轮都初始化vae_loss
        vae_losses = []
        discriminator_losses = []
        generator_losses = []

        for batch in range(x.shape[0] // batch_size):
            start = int(batch * batch_size)
            end = int(start + batch_size)
            samples = x[start:end]

            autoencoder_history = autoencoder.fit(x=samples, y=[samples, np.zeros([samples.shape[0],1]), np.zeros_like(samples)], epochs=30, batch_size=batch_size, validation_split=0.0, verbose=0)

            fake_latent = encoder.predict(samples)
            real_sample = s_gt[start:end]
            real_sample = np.random.normal(size=(batch_size, latent_dim))
            discriminator_input = np.concatenate((fake_latent, real_sample))
            discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=10, batch_size=batch_size, validation_split=0.0, verbose=0)
            generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=10, batch_size=batch_size, validation_split=0.0, verbose=0)
            autoencoder_losses.append(autoencoder_history.history["loss"])
            # vae_loss 每一轮都初始化用这个
            vae_losses.append(autoencoder_history.history["vae_loss"])
            # vae_loss 累积用这个
            # vae_losses = autoencoder_history.history["vae_loss"]

            discriminator_losses.append(discriminator_history.history["loss"])
            generator_losses.append(generator_history.history["loss"])
        # if (epoch + 1) % 2 == 0:
        print("\nEpoch {}/{} ".format(epoch, n_epochs, ))
        print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
        print("VAE Loss: {}".format(np.mean(vae_losses)))

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

    desc = "avae"



    output_path = 'C:/Users/10070/Desktop/HsU/Results'
    method_name = 'AVAeNet'
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
