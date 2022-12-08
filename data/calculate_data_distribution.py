import numpy as np
import os

import FID
import dataloader

mu, sigma = FID.find_distribution(dataloader.get_batch(0, int(dataloader.data_size / 100)))

dir = os.path.dirname(__file__)

mu_dir = os.path.join(dir, 'mu.npy')
sigma_dir = os.path.join(dir, 'sigma.npy')

open(mu_dir, "w").close()
open(sigma_dir, "w").close()

np.save(mu_dir, mu)
np.save(sigma_dir, sigma)