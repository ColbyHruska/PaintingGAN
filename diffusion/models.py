import keras
from keras import layers
from keras import activations
from keras_unet.models import custom_unet as unet
from keras.optimizers import Adam
from keras.losses import MSE

def define_noise_predictor(input_shape=(None, 64,64,3)):
    in_image = layers.Input(input_shape)
    in_time = layers.Input((1))

    n_nodes = input_shape[0] * input_shape[1]
    time_embedding = layers.Dense(n_nodes, activations.tanh)(in_time)
    time_embedding = layers.Reshape((input_shape[0], input_shape[1], 1))(time_embedding)

    hid = layers.Concatenate()([in_image, time_embedding])
    hid = unet((input_shape[0], input_shape[1], input_shape[2] + 1), input_shape[2], output_activation=activations.linear)(hid)

    model = keras.Model([in_image, in_time], hid)
    opt = Adam(learning_rate=0.01)
    model.compile(opt, MSE)

    return model
