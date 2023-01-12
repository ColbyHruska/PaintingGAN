#External
import numpy as np
from numpy import ones, zeros
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers, activations
from keras import models as M
from keras.optimizers import Adam
from PIL import Image
import sys
#Internal
from data import dataloader, FID, outputs
from gan import samples
from gan import models

max_epoch = 5000
store_img_iter = 100
display_stats_iter = 100
batch_size = 32

n_critic = 5

latent_dim = 128

save_path = os.path.join(os.path.dirname(__file__), "trained_models/gan")
def save_model(model):
	model.save(save_path)

def train(d_model, g_model, gan_model):
	sample_vector = samples.generate_latent_vectors(latent_dim, 16)

	n_batches = int(dataloader.data_size / batch_size)
	for epoch in range(max_epoch):
		for batch in range(n_batches):
			print(batch)
			real_images = dataloader.get_batch(batch * batch_size, batch_size)
			fake_images = samples.generate_fake_samples(g_model, latent_dim, batch_size)

			d_loss = 0

			for _ in range(n_critic):
				X = real_images
				Y = ones((batch_size,1)) * -1
				d_loss += d_model.train_on_batch(X, Y)

				X = fake_images
				Y = ones((batch_size,1))
				d_loss += d_model.train_on_batch(X, Y)

			d_loss /= n_critic * 2

			X = samples.generate_latent_vectors(latent_dim, batch_size * 2)
			Y = ones((batch_size * 2, 1)) * -1
			g_loss = gan_model.train_on_batch(X, Y)

			if (batch + 1) % display_stats_iter == 0:
				print(f"{epoch}: {batch}/{n_batches}) d_loss = {d_loss}, g_loss = {g_loss}, god: {g_loss / d_loss}")
				print(f"FID: {FID.calculate_fid(fake_images)}")
				im = g_model.predict(sample_vector)
			#	im = np.moveaxis(im, -1, 0)
			#	print(im.shape)
				#samples.save_plot(dataloader.get_batch(np.random.randint(low=0, high=dataloader.data_size - 16, size=1)[0], 16))
				outputs.save_plot(im)
				save_model(g_model)

def main():
	print(f"Dataset size: {dataloader.data_size:,}")
	pretrained = None
	if "resume" in sys.argv:
		pretrained = M.load_model(save_path)
		print("Resuming..")
	
	models.define_models(latent_dim, g_model=pretrained)
	d_model = models.discriminator
	g_model = models.generator
	gan_model = models.gan

	train(d_model, g_model, gan_model)

if __name__ == "__main__":
	main()