import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import os
import math

model = InceptionV3(weights="imagenet", include_top=False, pooling='avg', input_shape=(299,299,3))

def scale(imgs, shape):
	img_list = []
	for img in imgs:
		new_img = resize(img, shape, 0)
		img_list.append(new_img)
	return np.asarray(img_list)

def features(arr, preprocess=True):
	arr = scale(arr, (299,299,3))
	arr = arr.astype('float32')
	if preprocess:
		arr = preprocess_input(arr)

	distribution = model.predict(arr)
	return distribution

def find_distribution(arr, preprocess=True):
	distribution = features(arr, preprocess)

	mu = distribution.mean(axis=0)
	sigma = np.cov(distribution, rowvar=False)
	return mu, sigma

def frechet(mu1, sigma1, mu2, sigma2):
	sum_square_diff = np.sum((mu1 - mu2) ** 2)
	dot = sigma1.dot(sigma2)
	covmean = sqrtm(dot)

	if np.iscomplexobj(covmean):
		covmean = covmean.real

	d = sum_square_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
	return d

def calculate_fid(images1 : np.array, images2 : np.array, preprocess : bool = True):
	mu1, sigma1, = find_distribution(images1, preprocess)
	mu2, sigma2, = find_distribution(images2, preprocess)

	return frechet(mu1, sigma1, mu2, sigma2)

dir = os.path.dirname(__file__)
data_mu = np.load(os.path.join(dir, 'mu.npy'))
data_sigma = np.load(os.path.join(dir, 'sigma.npy'))

def calculate_fid(images : np.array, preprocess : bool = True):
	mu, sigma = find_distribution(images, preprocess)
	
	return frechet(data_mu, data_sigma, mu, sigma) 

def test():
	return frechet(data_mu, data_sigma, data_mu, data_sigma) 

inv = np.linalg.inv(data_sigma)
def likelihood(img, preproccess=True):
	feat = features(np.expand_dims(img, 0), preproccess)[0]

	diff = feat - data_mu

	exp = np.dot(inv, diff)
	exp = np.dot(diff, exp)
	l = -exp

	return l