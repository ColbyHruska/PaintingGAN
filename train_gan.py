#External
import numpy as np
from numpy import ones, zeros
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers
from keras import activations
from keras.optimizers import Adam
from PIL import Image
#Internal
from data import dataloader, FID, outputs
from gan import samples
from gan import models64 as models

print(f"Dataset size: {dataloader.data_size:,}")

max_epoch = 5000
store_img_iter = 100
display_stats_iter = 10
batch_size = 64

latent_dim = 100

def train(d_model, g_model, gan_model):
	sample_vector = samples.generate_latent_vectors(latent_dim, 16)

	n_batches = int(dataloader.data_size / batch_size)
	for epoch in range(max_epoch):
		for batch in range(n_batches):
			print(batch)
			real_images = dataloader.get_batch(batch * batch_size, batch_size)
			fake_images = samples.generate_fake_samples(g_model, latent_dim, batch_size)

			X = real_images
			Y = ones((batch_size,1))
			d_loss1, acc1 = d_model.train_on_batch(X, Y)

			X = fake_images
			Y = zeros((batch_size,1))
			d_loss2, acc2 = d_model.train_on_batch(X, Y)

			d_loss = (d_loss1 + d_loss2) / 2
			acc = (acc1 + acc2) / 2

			X = samples.generate_latent_vectors(latent_dim, batch_size * 2)
			Y = ones((batch_size * 2, 1))
			g_loss = gan_model.train_on_batch(X, Y)

			if (batch + 1) % display_stats_iter == 0:
				print(f"{epoch}: {batch}/{n_batches}) d_loss = {d_loss}, accuracy = {acc}, g_loss = {g_loss}")
				print(f"FID: {FID.calculate_fid(fake_images)}")
				im = g_model.predict(sample_vector)
			#	im = np.moveaxis(im, -1, 0)
			#	print(im.shape)
				#samples.save_plot(dataloader.get_batch(np.random.randint(low=0, high=dataloader.data_size - 16, size=1)[0], 16))
				outputs.save_plot(im)

models.define_models(latent_dim)

d_model = models.discriminator
g_model = models.generator
gan_model = models.gan

train(d_model, g_model, gan_model)