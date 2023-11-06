import os
import glob

import csv
import shutil
import numpy as np
import h5py

from matplotlib import pyplot as plt

merging_dir = "galaxy_zoo_merging/"
non_merging_dir = "galaxy_zoo_non_merging/"


merging_image_files = sorted(glob.glob(merging_dir+"*.jpg"))
non_merging_image_files = sorted(glob.glob(non_merging_dir+"*.jpg"))

image_data = np.empty((61578, 3, 424, 424), dtype=np.float32)
index = 0
for count, file in enumerate(merging_image_files):
  print(count, file)
  im = np.transpose(plt.imread(file), (2, 0, 1))
  print(im.shape)
  image_data[index,:,:,:] = im/255.0
  index += 1
  
for count, file in enumerate(non_merging_image_files):
  if (count%100 == 0):
    print(count, file)
  im = np.transpose(plt.imread(file), (2, 0, 1))
  # print(im.shape)
  image_data[index,:,:,:] = im/255.0
  index += 1
  
print(image_data.shape)
print(f"index={index}")
total_mem_size = image_data.size * image_data.itemsize
print(f"total_mem_size={total_mem_size}")

hf = h5py.File('galaxy_zoo_all.h5', 'w')
hf.create_dataset('images', data=image_data)
hf.close()
