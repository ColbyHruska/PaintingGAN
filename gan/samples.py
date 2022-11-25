import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
from matplotlib import pyplot

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
		dir = os.path.join(os.path.dirname(__file__), f"./out/")
		n = len(os.listdir(dir))
		im.save(os.path.join(dir, f"{n}.png"))

def save_plot(examples, n=4):
	dir = os.path.join(os.path.dirname(__file__), f"./out/")
	file = os.path.join(dir, f"{len(os.listdir(dir))}.png")
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i])
	pyplot.savefig(file, dpi = 159)
	pyplot.close('all')
	pyplot.close()