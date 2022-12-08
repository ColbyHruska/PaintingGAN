import numpy as np
import tensorflow as tf
from tensorflow import keras

def seed(seed):
	np.random.seed(seed)

def generate_latent_vectors(n_dim, n_samples):
	x = np.random.randn(n_dim * n_samples)
	x = x.reshape(n_samples, n_dim)
	return x

def generate_fake_samples(g_model, n_dim, n_samples):
	x_input = generate_latent_vectors(n_dim, n_samples)

	return g_model.predict([x_input])