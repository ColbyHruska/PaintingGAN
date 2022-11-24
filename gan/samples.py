import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

def seed(seed):
	np.random.seed(seed)

def generate_latent_vectors(n_dim, n_samples):
	x = np.random.randn(n_dim * n_samples)
	x = x.reshape(n_samples, n_dim)
	return x

def generate_fake_samples(g_model, n_dim, n_samples):
	x_input = generate_latent_vectors(n_dim, n_samples)
	return g_model.predict([x_input])

def save_image(arr):
	with Image.fromarray((arr * 255).astype(np.uint8)) as im:
		print(arr * 255)
		dir = os.path.join(os.path.dirname(__file__), f"./out/")
		n = len(os.listdir(dir))
		im.save(os.path.join(dir, f"{n}.png"))