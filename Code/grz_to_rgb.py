#%load_ext autoreload
#%autoreload 2

import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info
from ssl_legacysurvey.utils import to_rgb


from ssl_legacysurvey.data_loaders import datamodules # Pytorch dataloaders and datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations # Augmentations for training

from ssl_legacysurvey.data_analysis import dimensionality_reduction # PCA/UMAP functionality

grz_data_path = "decals_galaxy/images_npix152_000000000_003500000.h5"
rgb_data_path = "decals_galaxy_rgb_10000.h5"


grz = h5py.File(grz_data_path, 'r')

for k in grz:
    print(f"{k} shape:", grz[k].shape, grz[k].dtype)
    
# rgb = np.array(hf['images'][300000,:,:,:])
# print("DEBUG")
# print(hf['images'][300000,:,:,:])
# print("DEBUG")
# print(rgb)

file_rgb = h5py.File(rgb_data_path, 'w')
dst_size = list(grz['images'].shape)
dst_size[0]=10000
data_rgb = file_rgb.create_dataset('images', tuple(dst_size))
print("Processing ", dst_size)

#for k in range(grz['images'].shape[0]):
for k in range(dst_size[0]):
   rgb_k = to_rgb.dr2_rgb(grz['images'][k,:,:,:], ['g','r','z'])[::-1]
   #print(rgb_k.shape)
   #print(grz['images'][k,:,:,:].shape)
   data_rgb[k,:,:,:] = np.transpose(rgb_k, (2, 0, 1))
   #print(data_rgb[k,:,:,:].shape)
   
grz.close()
file_rgb.close()
