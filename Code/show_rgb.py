import os
import glob

import csv
import shutil
import numpy as np
import h5py

from matplotlib import pyplot as plt

# r = np.array(range(24))
# r = r.reshape(2,4, 3)
# print(r)
# print(r[::-1])


img_file = "galaxy_zoo_merging/999964.jpg"

rgb = plt.imread(img_file)
print(rgb/255)
#im = np.transpose(rgb, (2, 0, 1))
plt.imshow(rgb/255)
plt.axis('off')
plt.show()