import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import keras
import sys

from data import dataloader, FID, outputs
from diffusion import models
import training_sessions

max_epoch = 5000
store_img_iter = 300
display_stats_iter = 400
batch_size = 80

image_shape = (64, 64, 3)

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

def get_samples(n_samples):
    idx = np.random.randint(0, dataloader.data_size, n_samples)
    timestamps = []
    images = []
    noises = []
    for i in idx:
        timestamps.append(np.random.randint(0, timesteps))

        image, noise = forward_noise(dataloader.get_batch(i, 1)[0], timestamps[-1])
        images.append(image)
        noises.append(noise)
    
    return (np.array(images), np.array(timestamps)), np.array(noises)

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss

def ddpm(x_t, pred_noise, t, z=None):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

    var = np.take(beta, t)
    if z is None:
        z = np.random.normal(size=x_t.shape)

    return mean + (var ** .5) * z

def generate_images(n_images, model, seed=None, normals = None):
    nextseed = None
    if seed != None:
        nextseed = np.random.randint(1, 1e9)
        np.random.seed(seed)
    images = []
    for i in range(n_images):
        img = None
        if normals is None:
            img = generate_image(model)
        else:
            img = generate_image(model, normals[i])
        images.append(img)
    if nextseed != None:
        np.random.seed(nextseed)
    return images

def generate_image(model, normals = None):
    t = timesteps - 1
    img = np.random.normal(0, 1, image_shape)

    while t > 0:
        normal = None if normals is None else normals[t - 1]
        img = ddpm(img, model.call([tf.convert_to_tensor(np.array([img])), tf.convert_to_tensor(np.array([t]))])[0], t, normal)
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
    for epoch in range(max_epoch):
        n_batches = int(dataloader.data_size / batch_size)
        for batch in range(n_batches):
            #print(f"{epoch}) {batch}/{n_batches}")
            X, Y = get_samples(batch_size)

            loss = model.train_on_batch(X, Y)

            if (batch + 1) % display_stats_iter == 0:
                print(f"{epoch}: {batch}/{n_batches}) loss = {loss}")
                images = generate_images(4, model, seed=100)
                images = np.clip(np.array(images), -1, 1)
                print(f"FID: {FID.calculate_fid(images)}")
			#	im = np.moveaxis(im, -1, 0)
			#	print(im.shape)
				#samples.save_plot(dataloader.get_batch(np.random.randint(low=0, high=dataloader.data_size - 16, size=1)[0], 16))
                sess.save_plot(images, 2)
                sess.save()

def main():
    model = None
    group = training_sessions.SessionGroup("D")
    sess = None
    if "resume" in sys.argv:
        sess = group.TrainingSession(group.latest())
        model = sess.models["diff"]
    else:
        model = models.define_noise_predictor(image_shape)
        sess = group.TrainingSession({"diff" : model})
    train(model, sess)

if __name__ == "__main__":
    main()