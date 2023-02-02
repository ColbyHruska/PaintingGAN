import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import losses
import sys
import math

from data import dataloader, FID, outputs
from diffusion import models
import training_sessions

from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices("GPU"))

max_epoch = 5000
store_img_iter = 300
display_stats_iter = 400
batch_size = 48
norm_groups = 8
learning_rate = 1e-4

img_size = 64
n_channels = 3
image_shape = (img_size, img_size, n_channels)

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2

timesteps = 500
b1, b2 = 0.0001, 0.02
beta = np.cos(np.linspace(0, math.pi / 2, timesteps)) * (b2 - b1) + b1
alpha = 1 - beta
#print(alpha)
alpha_bar = np.cumprod(alpha, 0)
#alpha_bar = np.concatenate((np.array([1]), alpha_bar))
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1 - alpha_bar)

def forward_noise(x, t):
    noise = tf.random.normal(shape=x.shape)

    mean = sqrt_alpha_bar[t] * x
    noised_image = mean + one_minus_sqrt_alpha_bar[t] * noise

    return noised_image, noise

def maximum(*arr):
    x = arr[0]
    for i in arr:
        x = np.maximum(x, i)
    return x

def minimum(*arr):
    x = arr[0]
    for i in arr:
        x = np.minimum(x, i)
    return x

ROOT3 = math.sqrt(3)
def convert_colour(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    M = maximum(r, g, b)
    m = minimum(r, g, b)
    delta = M - m

    s = M + 1
    s = 2 * delta / s
    s = np.nan_to_num(s)

    d = np.sqrt(np.square(g - b) * 3 + np.square(2 * r - g - b))

    x = 2 * r - g - b / d
    y = ROOT3 * (g - b) / d

    x = np.nan_to_num(x) * s
    y = np.nan_to_num(y) * s

    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    z = np.expand_dims(M, -1)

    return np.concatenate((x, y, z), axis=-1)

def to_rgb(img):
    img = np.array(img)

    x = img[:, :, 0]
    y = img[:, :, 1]
    z = img[:, :, 2]

    s = np.sqrt(np.square(x) + np.square(y))

    m = -(s * (z + 1)) / 2 + z

    up = y >= 0
    down = np.logical_not(up)
    slope = y/x
    a1 = ((slope >= 0) & (slope < ROOT3))
    a3 = ((slope >= -ROOT3) & (slope < 0))
    a2 = (np.logical_not(a1 | a3))
    a4 = down & a1
    a5 = down & a2
    a6 = down & a3
    a1 = up & a1
    a2 = up & a2
    a3 = up & a3

    img_shape = x.shape
    r = np.zeros(img_shape)
    g = np.zeros(img_shape)
    b = np.zeros(img_shape)

    xy = x/y

    bi1 = 1 + xy * ROOT3
    bi2 = 2 - bi1

    r += a1 * z
    r += a6 * z
    r += a3 * m
    r += a4 * m
    r += a2 * (ROOT3 * (z - m) *  xy + z + m) / 2
    r += a5 * (ROOT3 * (m - z) *  xy + z + m) / 2

    g += a2 * z
    g += a3 * z
    g += a5 * m
    g += a6 * m
    g += a1 * ((2 * z - bi2 * m) / bi1)
    g += a4 * ((2 * m - bi2 * z) / bi1)

    b += a1 * m
    b += a2 * m
    b += a4 * z
    b += a5 * z
    b += a3 * ((2 * m - bi1 * z) / bi2)
    b += a6 * ((2 * z - bi1 * m) / bi2)

    r = np.expand_dims(r, -1)
    g = np.expand_dims(g, -1)
    b = np.expand_dims(b, -1)

    return np.concatenate((r, g, b), axis=-1)
def get_samples(n_samples):
    idx = tf.random.uniform([n_samples], 0, dataloader.data_size, dtype=tf.dtypes.int32)
    timestamps = tf.random.uniform([n_samples], 0, timesteps, dtype=tf.dtypes.int32)
    images = []
    noises = []
    for i in range(n_samples):
        image, noise = forward_noise(dataloader.get_batch(idx[i], 1)[0], timestamps[i])
        images.append(image)
        noises.append(noise)
    
    return (tf.convert_to_tensor(images), np.array(tf.expand_dims(timestamps, -1))), tf.convert_to_tensor(noises)

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss

def ddpm(x_t, pred_noise, t, generator):
    alpha_t = np.take(alpha, t)
    #print(alpha_t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / ((1 - alpha_t_bar) ** .5)
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)
    
    var = np.take(beta, t)
    if t == 0:
        z = tf.zeros(x_t.shape)
    else:
        z = generator.normal(size=x_t.shape)

    return mean + (var ** .5) * z

def generate_images(n_images, model, seed):
    gen = tf.random.Generator.from_seed(seed)
    return generate_images(n_images, model, gen)

def generate_images(n_images, model, gen):
    images = gen.normal((n_images, image_shape[0], image_shape[1], image_shape[2]))

    for t in reversed(range(timesteps)):
        noise = model.call([images, tf.convert_to_tensor(np.array([t] * n_images))])
        
        alpha_t = np.take(alpha, t)
        alpha_t_bar = np.take(alpha_bar, t)

        eps_coef = (1 - alpha_t) / ((1 - alpha_t_bar) ** .5)
        mean = (1 / (alpha_t ** .5)) * (images - eps_coef * noise)
    
        var = np.take(beta, t)

        if t == 0:
            z = tf.zeros(images.shape)
        else:
            z = gen.normal(shape=images.shape)
        
        images = mean + (var ** .5) * z
    images = tf.clip_by_value(images, -1, 1)
    return images

def save_model(model):
    model.save(os.path.join(os.path.dirname(__file__), "trained_models/diffusion"))

#sample_normals = []
#for i in range(16):
#    normals = []
#    for j in range(int(timesteps / 2)):
#        normals.append(np.random.normal(size=image_shape))
#    sample_normals.append(normals)

def train(model, sess):
    ma_loss = 0
    n = 0
    for epoch in range(max_epoch):
        n_batches = int(dataloader.data_size / batch_size)
        for batch in range(n_batches):
            #print(f"{epoch}) {batch}/{n_batches}")
            X, Y = get_samples(batch_size)
            
            #print(f"max: {np.max(np.reshape(X[0], (-1)))}")
            #print(f"X: {X}")
#            print(f"Y: {Y}")

            loss = model.train_on_batch(X, Y)
            ma_loss += loss
            n += 1

            if (batch % display_stats_iter == 0) and (batch != 0):
                print(f"{epoch}: {batch}/{n_batches}) loss = {loss}, ma_loss = {ma_loss / n}")
                ma_loss = 0; n = 0
                images = generate_images(4, model, 72)
                sess.save_plot(images, 2)
                #print(f"FID: {FID.calculate_fid(images)}")
			#	im = np.moveaxis(im, -1, 0)
			#	print(im.shape)
				#samples.save_plot(dataloader.get_batch(np.random.randint(low=0, high=dataloader.data_size - 16, size=1)[0], 16))
                sess.save()

def main():
    model = None
    group = training_sessions.SessionGroup("D")
    sess = None
    if "resume" in sys.argv:
        sess = group.load_sess(path=group.latest())
        model = sess.models["diff"]
    else:
        model = models.define_noise_predictor(image_shape, learning_rate, n_layers=6)
        sess = group.new_sess(models={"diff" : model})
    train(model, sess)

if __name__ == "__main__":
    main()