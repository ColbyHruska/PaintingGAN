#External
import numpy as np
from numpy import ones, zeros
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers
from keras.optimizers import SGD
from PIL import Image
#Internal
import dataloader
import samples

max_epoch = 5000
store_img_iter = 100
display_stats_iter = 10
batch_size = 20

latent_dim = 128

np.random.seed(245)

def generate_latent_vectors(n_dim, n_samples):
	x = np.random.randn(n_dim * n_samples)
	x = x.reshape(n_samples, n_dim)
	return x

def generate_fake_samples(g_model, n_dim, n_samples):
	x_input = generate_latent_vectors(n_dim, n_samples)
	return g_model.predict([x_input])

def save_image(arr):
	with Image.fromarray((arr * 255).astype(np.uint8)) as im:
		n = len(os.listdir("./out/"))
		im.save(f"./out/{n}.png")

def define_generator(latent_dim):
	relu_alpha = 0

	n_nodes = 32 ** 2
	n_nodes *= 256
	in_lat = keras.Input(shape=(latent_dim,))
	gen = layers.Dense(n_nodes)(in_lat)
	gen = layers.Reshape((32, 32, 256))(gen)
	gen = layers.Conv2D(128, 4, padding='same')(gen)

	gen = layers.UpSampling2D()(gen)
	gen = layers.Conv2D(64, 4, padding='same')(gen)
	gen = layers.LeakyReLU(relu_alpha)(gen)

	gen = layers.UpSampling2D()(gen)
	gen = layers.Conv2D(32, 4, padding='same')(gen)
	gen = layers.LeakyReLU(relu_alpha)(gen)

	gen = layers.Conv2D(10, 4, 2, padding='same')(gen)
	gen = layers.UpSampling2D()(gen)
	gen = layers.LeakyReLU(relu_alpha)(gen)
	gen = layers.Conv2D(10, 4, padding='same')(gen)
	gen = layers.LeakyReLU(relu_alpha)(gen)
	gen = layers.Conv2D(3, 4, padding='same')(gen)
	out = layers.Conv2D(3, 4, padding='same')(gen)

	model = keras.Model([in_lat], out)
	return model

def define_discriminator(in_shape=(128,128,3)):
	relu_alpha = 0.3

	in_image = layers.Input(shape=in_shape)
	dis = layers.Conv2D(32, 3, 2, padding='same')(in_image)
	dis = layers.LeakyReLU(relu_alpha)(dis)

	dis = layers.Conv2D(64, 4, 2, padding='same')(dis)
	dis = layers.Conv2D(64, 5, 2, padding='same')(dis)
	dis = layers.LeakyReLU(relu_alpha)(dis)

	dis = layers.Conv2D(128, 5, 2, padding='same')(dis)
	dis = layers.LeakyReLU(relu_alpha)(dis)
	dis = layers.Conv2D(128, 5, 2, padding='same')(dis)
	dis = layers.LeakyReLU(relu_alpha)(dis)

	dis = layers.Flatten()(dis)
	dis = layers.Dropout(0.5)(dis)
	out = layers.Dense(1)(dis)

	model = keras.Model([in_image], out)
	opt = SGD(learning_rate=0.0002)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_gan(g_model, d_model):
	d_model.trainable = False

	gen_noise = g_model.input
	gen_output = g_model.output
	gan_output = d_model([gen_output])

	model = keras.Model([gen_noise], gan_output)
	opt = SGD(learning_rate=0.0002)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model


def train(d_model, g_model, gan_model):
	sample_vector = generate_latent_vectors(latent_dim, 1)

	n_batches = int(dataloader.data_size / batch_size)
	for epoch in range(max_epoch):
		for batch in range(n_batches):
			real_images = dataloader.get_batch(batch * batch_size, batch_size)
			fake_images = generate_fake_samples(g_model, latent_dim, batch_size)

			X = [np.concatenate((real_images, fake_images))]
			Y = np.concatenate((ones(batch_size), zeros(batch_size)))
			d_loss, acc = d_model.train_on_batch(X, Y)

			X = [generate_latent_vectors(latent_dim, batch_size * 2)]
			Y = ones(batch_size * 2)
			g_loss = gan_model.train_on_batch(X, Y)

			if (batch + 1) % display_stats_iter == 0:
				print(f"{epoch}: {batch}/{n_batches}) d_loss = {d_loss}, accuracy = {acc}, g_loss = {g_loss}")
				im = g_model.predict(sample_vector)[0]
			#	im = np.moveaxis(im, -1, 0)
				print(im.shape)
				save_image(im)

d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)

train(d_model, g_model, gan_model)