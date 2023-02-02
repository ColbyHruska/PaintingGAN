import keras
from keras import Model
import numpy as np
import os
import tensorflow as tf
from data import outputs
from gan import samples

n_dim = 100
model : Model = keras.models.load_model(os.path.join(os.path.dirname(__file__), "trained_models/gan"))

n = 800
for i in range(n):
    vector = samples.generate_latent_vectors(n_dim, 1)
    img = model.predict(vector)[0]
    outputs.save_image(img, os.path.join(os.path.dirname(__file__), "data/generated"))