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

def generate_images(n_images, model, generator):
    images = []
    for i in range(n_images):
        img = None
        img = generate_image(model, generator)
        images.append(img)
    return images

def generate_image(model, generator):
    t = timesteps - 1
    img = np.random.normal(0, 1, image_shape)

    while t >= 0:
        img = ddpm(img, model.call([tf.convert_to_tensor(np.array([img])), tf.convert_to_tensor(np.array([t]))])[0], t, generator)
        t -= 1
    return img

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
                sample_gen = np.random.default_rng(seed=72)
                images = generate_images(4, model, sample_gen)
                images = np.clip(np.array(images), -1, 1)
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