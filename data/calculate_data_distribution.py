import numpy as np
import os

import features
import dataloader

batch_size = 500
feature_arr = features.batch_features(dataloader.get_random_batch(batch_size), False)
rem = 10000
rem -= batch_size
i = 0
while rem > 0:
    n = min(rem, batch_size)
    feature_arr = np.concatenate((feature_arr, features.batch_features(dataloader.get_random_batch(n), False)), axis=0)
    i += n
    rem -= n
    print(f"{i}/{dataloader.data_size}")

mu, sigma = features.feature_distribution(feature_arr)

dir = os.path.dirname(__file__)
mu_dir = os.path.join(dir, 'mu.npy')
sigma_dir = os.path.join(dir, 'sigma.npy')

def try_del(dir):
    if os.path.exists(dir):
        os.remove(dir)

try_del(mu_dir)
try_del(sigma_dir)

np.save(mu_dir, mu)
np.save(sigma_dir, sigma)