from tensorflow import keras
from keras import layers
from keras import activations
from keras.optimizers import Adam

discriminator, generator, gan = None, None, None

def define_generator(latent_dim):
	relu_alpha = 0.3
	momentum = 0.9

	n_nodes = 1 ** 2
	n_nodes *= 64
	in_lat = keras.Input(shape=(latent_dim,))
	gen = layers.Dense(n_nodes)(in_lat)

	gen = layers.Reshape((1, 1, 64))(gen)

	for _ in range(3):
		gen = layers.BatchNormalization(momentum=momentum)(gen)
		gen = layers.UpSampling2D()(gen)
		gen = layers.Conv2D(64, 3, 1, padding='same')(gen)
		gen = layers.Conv2DTranspose(64, 3, 2, padding='same')(gen)

	out = layers.Conv2D(3, 3, padding='same')(gen)
	out = layers.Activation(activations.tanh)(out)

	model = keras.Model([in_lat], out)
	return model

def define_discriminator(in_shape=(64,64,3)):
	relu_alpha = 0.2
	momentum = 0.9

	in_image = layers.Input(shape=in_shape)
	dis = layers.Conv2D(64, 3, 2, padding='same')(in_image)

	for _ in range(3):
		dis = layers.Conv2D(64, 3, 1, padding='same')(dis)
		dis = layers.LeakyReLU(relu_alpha)(dis)
		dis = layers.Dropout(0.5)(dis)

	out = layers.Flatten()(dis)
	out = layers.Dense(1, activation='sigmoid')(out)

	model = keras.Model([in_image], out)
	opt = Adam(learning_rate=0.0002)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def define_gan(g_model, d_model):
	d_model.trainable = False

	gen_noise = g_model.input
	gen_output = g_model.output
	gan_output = d_model([gen_output])

	model = keras.Model([gen_noise], gan_output)
	opt = Adam(learning_rate=0.0002)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def define_models(latent_dim, in_shape=None):
	global discriminator, generator, gan
	discriminator = define_discriminator(in_shape) if in_shape != None else define_discriminator()
	generator = define_generator(latent_dim)
	gan = define_gan(generator, discriminator)
	return (discriminator, generator, gan)