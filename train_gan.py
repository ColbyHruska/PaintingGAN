#External
import numpy as np
from numpy import ones
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import sys

#Internal
from data import dataloader
from gan import samples
from gan import models
import training_sessions

max_epoch = 5000
store_img_iter = 100
display_stats_iter = 100
batch_size = 32

n_critic = 4

latent_dim = 128

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
	print(f"n_batches = {n_batches}")
	for epoch in range(max_epoch):
		for batch in range(n_batches):
			X = samples.generate_latent_vectors(latent_dim, batch_size * 2)
			Y = -ones((batch_size * 2, 1))
			g_loss = gan_model.train_on_batch(X, Y)
			del X
			del Y

			d_loss = 0

			Y = -ones((batch_size, 1))
			for _ in range(n_critic):
				X = dataloader.get_random_batch(batch_size)
				d_loss += d_model.train_on_batch(X, Y)
				del X
				Y = -Y

				X = samples.generate_fake_samples(g_model, latent_dim, batch_size)
				d_loss += d_model.train_on_batch(X, Y)
				Y = -Y
			del Y

			d_loss /= n_critic * 2

			if (batch + 1) % display_stats_iter == 0:
				print(f"{epoch}: {batch}/{n_batches}) d_loss = {d_loss}, g_loss = {g_loss}")
				im = g_model.predict(sample_vector)
				sess.save_plot(im)
				sess.save()
		K.clear_session()

def main():
	print(tf.config.list_physical_devices("GPU"))

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