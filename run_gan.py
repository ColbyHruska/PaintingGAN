import keras
import keras.backend as K
from keras import Model
import numpy as np
import os
import tensorflow as tf
from data import outputs
from gan import samples
import training_sessions
from keras.utils.generic_utils import get_custom_objects

def w_loss(y_pred, y_true):
	return K.mean(y_pred * y_true)

get_custom_objects().update({"w_loss" : w_loss})

group = training_sessions.SessionGroup("Gan")

sess = group.load_sess(group.latest())

n_dim = 128
model : Model = sess.models["generator"]

n = 2000
for i in range(n):
    vector = samples.generate_latent_vectors(n_dim, 1)
    img = model.predict(vector)[0]
    outputs.save_image(img, os.path.join(os.path.dirname(__file__), "data/generated"))