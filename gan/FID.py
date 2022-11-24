import numpy as np
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

def scale(imgs, shape):
	img_list = []
	for img in imgs:
		new_img = resize(img, shape, 0)
		img_list.append(new_img)
	return np.asarray(img_list)


def calculate_fid(images1, images2, preprocess=True):
	if preprocess:
		images1 = images1.astype('float32')
		images2 = images2.astype('float32')

		images1 = scale(images1, (299,299,3))
		images2 = scale(images2, (299,299,3))

		images1 = preprocess_input(images1)
		images2 = preprocess_input(images2)

	distribution1 = model.predict(images1)
	distribution2 = model.predict(images2)

	mu1, mu2 = distribution1.mean(axis=0), distribution2.mean(axis=0)
	sigma1, sigma2 = np.cov(distribution1), np.cov(distribution2)

	sum_square_diff = np.sum((mu1 - mu2) ** 2)
	covmean = sqrtm(sigma1.dot(sigma2))

	if np.iscomplexobj(covmean):
		covmean = covmean.real

	fid = sum_square_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
	return fid

