import train_diff
import os
import tensorflow as tf
from data import outputs
import numpy as np
import math
from PIL import Image
from data import FID
import training_sessions
from data import dataloader
from data import features
group = training_sessions.SessionGroup("D")
sess = group.load_sess("2023-1-30-15-50-12", True)
model = sess.models["diff"]

out_path = os.path.join(os.path.dirname(__file__), "data/generated")
n_images = 2000
seed = 65478
gen = tf.random.Generator.from_seed(seed)
batch_size = 128
rem = n_images
while rem > 0:
    n = min(rem, batch_size)
#    imgs = np.array(train_diff.generate_images(n, model, gen))
    imgs = np.random.uniform(0, 1, (n, 64, 64, 3))
#    imgs = dataloader.get_random_batch(n) 
    for i in range(n):
        outputs.save_image(imgs[i], out_path)
    rem -= n

count = 2000
images = []
files = os.listdir(out_path)

feature_list = []

i = 0
for file in files[:min(len(files), count)]:
    with Image.open(os.path.join(out_path, file)) as img:
        images.append(np.array(img))
    i += 1
    if i == 100:
        feature_list.append(features.batch_features(np.array(images)))
        images = []
        i = 0
mu, sigma = features.feature_distribution(np.concatenate(feature_list, axis = 0))

print(f"FID: {FID.frechet(mu, sigma, FID.data_mu, FID.data_sigma)}")