import tensorflow as tf
import numpy as np

import models

max_epoch = 5000
store_img_iter = 100
display_stats_iter = 10
batch_size = 64

timesteps = 200
beta = np.linspace(0.0001, 0.02, timesteps)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1]), alpha_bar))
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1 - alpha_bar)

def forward_noise(x, t):
    noise = np.random.normal(size=x.shape)

    mean = sqrt_alpha_bar[t] * x
    noised_image = mean + one_minus_sqrt_alpha_bar[t] * noise

    return noised_image, noise

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss

