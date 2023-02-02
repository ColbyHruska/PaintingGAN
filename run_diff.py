import train_diff
import os
import tensorflow as tf
from data import outputs
import numpy as np
import math
from PIL import Image
from data import FID
import training_sessions
group = training_sessions.SessionGroup("D")
sess = group.load_sess("2023-1-30-15-50-12", True)
model = sess.models["diff"]

out_path = os.path.join(os.path.dirname(__file__), "data/generated")
n_images = 10000
seed = 65478
gen = tf.random.Generator.from_seed(seed)
batch_size = 128
rem = n_images
while rem > 0:
    n = min(rem, batch_size)
    imgs = np.array(train_diff.generate_images(n, model, gen))
    for i in range(n):
        outputs.save_image(imgs[i], out_path)
    rem -= n

images = []
for file in os.listdir(out_path):
    with Image.open(os.path.join(out_path, file)) as img:
        images.append(2 * np.array(img) / 255 - 1)
print(f"FID: {FID.calculate_fid(np.array(images), False)}")