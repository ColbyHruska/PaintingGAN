from tensorflow import keras
from keras import layers
from keras import activations
from keras.optimizers import RMSprop
import keras.backend as K
from keras.constraints import Constraint

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
	relu_alpha = 0.3
	momentum = 0.9
	
	img_size = 64
	channels = 3
	n_layers = 3
	up = 3
	kernals = 64

	start_size = img_size // (2 ** up)
	n_nodes = start_size ** 2
	n_nodes *= kernals
	in_lat = keras.Input(shape=(latent_dim,))
	gen = layers.Dense(n_nodes)(in_lat)

	gen = layers.Reshape((start_size, start_size, kernals))(gen)

	for _ in range(n_layers):
		#gen = layers.BatchNormalization(momentum=momentum)(gen)
		if up > 0:
			gen = layers.UpSampling2D()(gen)
			up -= 1
		gen = layers.Conv2DTranspose(kernals, 4, padding='same')(gen)
		#gen = layers.Conv2D(kernals, 5, 1, padding='same')(gen)
		gen = layers.LeakyReLU(relu_alpha)(gen)

	out = layers.Conv2D(channels, 4, padding='same')(gen)
	out = layers.Activation(activations.tanh)(out)

	model = keras.Model([in_lat], out)
	model.compile()
	return model

def define_discriminator(in_shape=(64,64,3)):
	relu_alpha = 0.2
	momentum = 0.9
	const = ClipConstraint(0.01)
	
	n_layers = 2
	kernals = 64

	in_image = layers.Input(shape=in_shape)
	dis = layers.Conv2D(kernals, 4, 2, padding='same', kernel_constraint=const)(in_image)

	for _ in range(n_layers):
		dis = layers.Conv2D(kernals, 4, 1, padding='same', kernel_constraint=const)(dis)
		#dis = layers.BatchNormalization(momentum=momentum)(dis)
		dis = layers.LeakyReLU(relu_alpha)(dis)
		#dis = layers.Dropout(0.5)(dis)

	out = layers.Flatten()(dis)
	out = layers.Dense(1)(out)

	model = keras.Model([in_image], out)
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=w_loss, optimizer=opt)
	return model

def define_gan(g_model, d_model):
	d_model.trainable = False

	gen_noise = g_model.input
	gen_output = g_model.output
	gan_output = d_model([gen_output])

	model = keras.Model([gen_noise], gan_output)
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=w_loss, optimizer=opt)

	return model

def define_models(latent_dim, in_shape=None, g_model=None):
	global discriminator, generator, gan
	discriminator = define_discriminator(in_shape) if in_shape != None else define_discriminator()
	if g_model == None:
		generator = define_generator(latent_dim)
	else:
		generator = g_model
	gan = define_gan(generator, discriminator)
	return (discriminator, generator, gan)