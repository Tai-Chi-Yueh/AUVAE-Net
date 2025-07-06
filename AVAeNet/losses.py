from keras import backend as K
import tensorflow as tf


def SAD(y_true, y_pred):
    y_true2 = tf.math.l2_normalize(y_true, axis=-1)
    y_pred2 = tf.math.l2_normalize(y_pred, axis=-1)
    A = tf.keras.backend.mean(y_true2 * y_pred2)
    sad = tf.math.acos(A)
    return sad


def normSAD(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    sad = SAD(y_true, y_pred)
    return 0.008 * mse + 1.0 * sad


def normMSE(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred))
    return mse


def normSAD2(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    sad = SAD(y_true, y_pred)
    return 0.005 * mse + 0.75 * sad


# 实现重参数化技巧
# 通过从各向同性单位高斯分布中采样实现重参数化技巧。
# Reparameterization trick by sampling from an isotropic unit Gaussian.
# Arguments 参数 args (tensor张量): mean and log of variance of Q(z|X) Q(z|X)的均值和对数方差
# Returns 返回值 z (tensor张量): sampled latent vector 采样得到的潜在向量
def resampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# vae_loss组合重构损失和KL散度
# VAE loss = reconstruction loss + KL divergence
def vae_loss(args):
    inputs=args[0]
    outputs=args[1]
    z_mean=args[2]
    z_log_var=args[3]
    reconstruction_loss = normSAD(inputs, outputs)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return reconstruction_loss + kl_loss