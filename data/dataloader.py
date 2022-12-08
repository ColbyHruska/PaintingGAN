import os
import numpy as np
from PIL import Image
import numpy as np

data_path = os.path.join(os.path.dirname(__file__),"images/")

def find_data_size():
	return len(os.listdir(data_path)) - 1 #data folder should always contain a number of images and a singe 'readme' file as a placeholder

data_size = find_data_size()

def get_batch(start : int, size : int):
	batch = []
	for i in range(start, start + size):
		with Image.open(data_path + f"{i}.png") as im:
			batch.append((np.array(im).astype('float32') - 127.5) / 127.5)
	return np.array(batch)