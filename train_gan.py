#External
import numpy as np
import gc
from numpy import ones, zeros
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers, activations
from keras import models as M
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from PIL import Image
import sys
import random
#Internal
from data import dataloader, FID, outputs
from gan import samples
from gan import models
import training_sessions

max_epoch = 5000
store_img_iter = 100
display_stats_iter = 100
batch_size = 32

n_critic = 4

latent_dim = 128

save_path = os.path.join(os.path.dirname(__file__), "trained_models/gan")
def save_model(model):
	model.save(save_path)

def mix(x, y, idx):
	concat = np.concatenate((x, y), axis=0)
	n = concat.shape[0]
	out = np.copy(concat)
	for i in range(n):
		j = idx[i]
		out[i] = concat[j].copy()
	del concat
	return out

def train(d_model, g_model, gan_model, sess):
	sample_vector = samples.generate_latent_vectors(latent_dim, 16)

	n_batches = int(dataloader.data_size / batch_size)
	for epoch in range(max_epoch):
		for batch in range(n_batches):
			#print(batch)

			X = samples.generate_latent_vectors(latent_dim, batch_size * 2)
			Y = ones((batch_size * 2, 1)) * -1
			g_loss = gan_model.train_on_batch(X, Y)

			d_loss = 0

			for _ in range(n_critic):
				real_images = dataloader.get_random_batch(batch_size)
				fake_images = samples.generate_fake_samples(g_model, latent_dim, batch_size)

				X = np.concatenate((real_images, fake_images))
				Y = np.concatenate((-ones((batch_size, 1)), ones((batch_size, 1))))

				d_loss += d_model.train_on_batch(X, Y)

			d_loss /= n_critic

			if (batch + 1) % display_stats_iter == 0:
				print(f"{epoch}: {batch}/{n_batches}) d_loss = {d_loss}, g_loss = {g_loss}, god: {g_loss / d_loss}")
				#print(f"FID: {FID.calculate_fid(fake_images)}")
				im = g_model.predict(sample_vector)
				sess.save_plot(im)
				sess.save()
		gc.collect()

def main():
	print(f"Dataset size: {dataloader.data_size:,}")
	group = training_sessions.SessionGroup("Gan")
	pretrained = None
	sess = None
	if "resume" in sys.argv:
		get_custom_objects().update({"w_loss" : models.w_loss})
		sess = group.load_sess(group.latest())
		pre_g = sess.models["generator"]
		pre_d = sess.models["discriminator"]
		print("Resuming..")
		models.load_models(latent_dim, (64, 64, 3), pre_g, pre_d)
	else:
		models.define_models(latent_dim, (64, 64, 3))
		sess = group.new_sess({"generator" : models.generator, "discriminator" : models.discriminator})

	d_model = models.discriminator
	g_model = models.generator
	gan_model = models.gan

	train(d_model, g_model, gan_model, sess)

if __name__ == "__main__":
	main()