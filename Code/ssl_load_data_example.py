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

from ssl_legacysurvey.data_loaders import datamodules # Pytorch dataloaders and datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations # Augmentations for training

from ssl_legacysurvey.data_analysis import dimensionality_reduction # PCA/UMAP functionality

data_path = "merging.h5"

plt.rcParams['figure.figsize']        = 8., 6.
plt.rcParams['axes.labelsize']        = 18
plt.rcParams['axes.titlesize']        = 18
plt.rcParams['xtick.top']             = True
plt.rcParams['xtick.bottom']          = True
plt.rcParams['ytick.left']            = True
plt.rcParams['ytick.right']           = True
plt.rcParams['xtick.direction']       = 'in'
plt.rcParams['ytick.direction']       = 'in'
plt.rcParams['xtick.labelsize']       = 18
plt.rcParams['ytick.labelsize']       = 18
plt.rcParams['xtick.major.pad']       = 6.
plt.rcParams['xtick.minor.pad']       = 6.
plt.rcParams['ytick.major.pad']       = 6.
plt.rcParams['ytick.minor.pad']       = 6.
plt.rcParams['xtick.major.size']      = 6. # major tick size in points
plt.rcParams['xtick.minor.size']      = 3. # minor tick size in points
plt.rcParams['ytick.major.size']      = 6. # major tick size in points
plt.rcParams['ytick.minor.size']      = 3. # minor tick size in points
plt.rcParams['text.usetex']           = False
plt.rcParams['font.family']           = 'serif'
plt.rcParams['font.size']             = 18

DDL = load_data.DecalsDataLoader(image_dir=data_path)

gals = DDL.get_data(-1, fields=DDL.fields_available) # -1 to load all galaxies

print("Available keys & data shapes:")
for k in gals:
    print(f"{k} shape:", gals[k].shape, gals[k].dtype)
print(gals['images'][0,:,:,:])
print(np.sum(gals['images'][0,:,:,:]))
# Display some pretty galaxy pictures
# nplt = 64
# ncol = 8
# npix_show = 152

# # Display images in grid, with ra/dec for each galaxy. Colors are after grb transformation, not raw data
# plt_tools.show_galaxies(gals['images'], gals['ra'], gals['dec'], 
#                         nx=ncol, nplt=nplt, npix_show=npix_show)

# plt.show()
