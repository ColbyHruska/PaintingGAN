import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations
from keras.optimizers import RMSprop, Adam, SGD
import keras.backend as K
from keras.constraints import Constraint
import random

discriminator, generator, gan = None, None, None

def w_loss(y_pred, y_true):
	return K.mean(y_pred * y_true)

class ClipConstraint(Constraint):
	def __init__(self, clip_value):
		self.clip_value = clip_value
	
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)
	
	def get_config(self):
		return {'clip_value': self.clip_value}

def define_generator(latent_dim):
	init = keras.initializers.RandomNormal; stddev=0.02
	
	relu_alpha = 0.2
	momentum = 0.8
	
	img_size = 64
	channels = 3
	n_layers = 4
	up = 5
	kernals = 256
	tran = 5

	start_size = img_size // (2 ** (up - 1))
	n_nodes = start_size ** 2
	n_nodes *= kernals
	start_shape = (start_size, start_size, kernals)
	in_lat = keras.Input(shape=(latent_dim,))
	gen = layers.Dense(n_nodes)(in_lat)
	gen = layers.Reshape(start_shape)(gen)
	
	for _ in range(n_layers):
		if tran > 0:
			stride = 2 if up > 0 else 1 
			gen = layers.Conv2DTranspose(kernals, 4, stride, padding='same', use_bias=False, kernel_initializer=init(stddev=stddev))(gen)
			tran -= 1
		else:
			gen = layers.Conv2D(kernals, 4, padding='same', kernel_initializer=init(stddev=stddev))(gen)
			if up > 0:
				gen = layers.UpSampling2D()(gen)
		kernals /= 2		
		up -= 1
		gen = layers.LeakyReLU(alpha=relu_alpha)(gen)

	out = layers.Conv2D(channels, 4, padding='same', kernel_initializer=init(stddev=stddev))(gen)
	out = layers.Activation(activations.tanh)(out)

	model = keras.Model([in_lat], out)
	model.compile()
	return model

def define_discriminator(in_shape=(64,64,3)):
	init = keras.initializers.RandomNormal

	relu_alpha = 0.2
	momentum = 0.8
	const = ClipConstraint(0.01)
	
	n_layers = 4
	kernals = 64

	in_image = layers.Input(shape=in_shape)
	dis = layers.Conv2D(3, 4, padding='same', kernel_constraint=const, kernel_initializer=init(stddev=0.02))(in_image)
	dis = layers.LeakyReLU(relu_alpha)(dis)
	dis = layers.Conv2D(kernals, 4, padding='same', kernel_constraint=const, kernel_initializer=init(stddev=0.02))(dis)

	for _ in range(n_layers):
		kernals *= 2
		dis = layers.LeakyReLU(relu_alpha)(dis)
		dis = layers.Conv2D(kernals, 4, 2, padding='same', kernel_constraint=const, kernel_initializer=init(stddev=0.02))(dis)
		#dis = layers.BatchNormalizationV1()(dis)
	dis = layers.LeakyReLU(relu_alpha)(dis)
	out = layers.Flatten()(dis)
	out = layers.Dense(1)(out)

	model = keras.Model([in_image], out)
	opt = RMSprop(learning_rate=5e-5)
	model.compile(loss=w_loss, optimizer=opt)
	return model

def define_gan(g_model, d_model):
	d_model.trainable = False

	gen_noise = g_model.input
	gen_output = g_model.output
	gan_output = d_model([gen_output])

	model = keras.Model([gen_noise], gan_output)
	opt = RMSprop(learning_rate=5e-5)
	model.compile(loss=w_loss, optimizer=opt)

	return model

def define_models(latent_dim, in_shape=None,):
	discriminator = define_discriminator(in_shape) if in_shape != None else define_discriminator()
	generator = define_generator(latent_dim)
	return load_models(latent_dim=latent_dim, in_shape=in_shape, g_model=generator, d_model=discriminator)	

def load_models(latent_dim, in_shape, g_model, d_model):
	global discriminator, generator, gan

	generator = g_model
	discriminator = d_model
	gan = define_gan(g_model, d_model)
	return (discriminator, generator, gan)
