import os
import glob

import csv
import shutil
import numpy as np
import h5py

from matplotlib import pyplot as plt

img_dir = "galaxy_zoo_merging/"

image_files = sorted(glob.glob(img_dir+"*.jpg"))
image_data = np.empty((1022, 3, 424, 424), dtype=np.float32)
for count, file in enumerate(image_files):
  print(count, file)
  im = np.transpose(plt.imread(file), (2, 0, 1))
  print(im.shape)
  image_data[count,:,:,:] = im/255.0
  
print(image_data.shape)

hf = h5py.File('merging.h5', 'w')
hf.create_dataset('images', data=image_data)
hf.close()


#plt.imshow(im)
#plt.axis('off')

#plt.show()


# with open(label_file) as infile:
#     reader = csv.reader(infile)
#     next(reader) # Skip the first row
#     for row in reader:
#       if (float(row[24]) > 0.5):
#         source_file = os.path.join(img_dir, row[0] + ".jpg")
#         dst_file = os.path.join(dst_dir, row[0] + ".jpg")
#         shutil.copyfile(source_file, dst_file)

