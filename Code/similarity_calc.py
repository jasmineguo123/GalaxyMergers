import os
import argparse

import h5py
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

from ssl_legacysurvey.utils import load_data # Loading galaxy catalogue and image data from hdf5 file(s)
from ssl_legacysurvey.utils import plotting_tools as plt_tools # Plotting images or catalogue info
from ssl_legacysurvey.utils import to_rgb


from ssl_legacysurvey.data_loaders import datamodules # Pytorch dataloaders and datamodules
from ssl_legacysurvey.data_loaders import decals_augmentations # Augmentations for training

from ssl_legacysurvey.data_analysis import dimensionality_reduction # PCA/UMAP functionality

merging_img_path = "merging.h5"
merging_shrink_img_path = "merging_shrinked.h5"
decals_img_path = "decals_galaxy_rgb.h5"
merging_rep_path = "result/merging_feature/compiled/model_outputs_000000000_000001022.npy"
decals_rep_path = "result/decals_feature/compiled/model_outputs_000000000_003500000.npy"
merging_decals_dist_path = "merging_decals_distance.h5"
merging_decals_dist_naive_path = "merging_decals_distance_naive.h5"

galaxy_zoo_all_img_path = "galaxy_zoo_all.h5"
galaxy_zoo_all_rep_path = "result/galaxy_zoo_all_feature/compiled/model_outputs_000000000_000061578.npy"
galaxy_zoo_all_dist_path = "galaxy_zoo_all_distance.h5"
galaxy_zoo_all_naive_dist_path = "galaxy_zoo_all_distance_naive.h5"


def calculate_galaxy_zoo_dist_naive():
  galaxy_zoo_all_img = h5py.File(galaxy_zoo_all_img_path, 'r')
  
  file_dist = h5py.File(galaxy_zoo_all_naive_dist_path, 'w')

  print(galaxy_zoo_all_img['images'].shape)
  # query_num = merging_img['images'].shape[0]
  query_num = 700
  total_num = galaxy_zoo_all_img['images'].shape[0]
  dist_data = file_dist.create_dataset('dist', (query_num, total_num - query_num))
  
      
  for index in range(query_num):
    print(f"Processing {index}-th image")
    query_data = galaxy_zoo_all_img['images'][index,:,:,:].flatten()
    query_data = query_data/np.linalg.norm(query_data)
    for i in range(total_num - query_num):
      second_data = galaxy_zoo_all_img['images'][i + query_num,:,:,:].flatten()
      second_data = second_data/np.linalg.norm(second_data)
      dist_data[index, i] = np.inner(query_data, second_data)
      
  file_dist.close()

def calculate_dist_galaxy_zoo():
  galaxy_zoo_all_no_used = np.load(galaxy_zoo_all_rep_path, mmap_mode='r')
  # normalize
  
  galaxy_zoo_all = galaxy_zoo_all_no_used/np.linalg.norm(galaxy_zoo_all_no_used, axis=1, keepdims=True)
  print(galaxy_zoo_all.shape)
  
  training_no = 700
  total_num = galaxy_zoo_all.shape[0]

  file_dist = h5py.File(galaxy_zoo_all_dist_path, 'w')
  dist_data = file_dist.create_dataset('dist', (training_no, total_num))


  for index in range(training_no):
    print(f"Processing {index}-th image")
    dist_data[index,:] = np.inner(galaxy_zoo_all[index,:], galaxy_zoo_all)
    # summary_chunk = np.maximum(dist_chunk, summary_chunk)

  print(dist_data[699,:])
  file_dist.close()


def calculate_dist():
  merging_rep_no_used = np.load(merging_rep_path, mmap_mode='r')
  decals_rep_no_used = np.load(decals_rep_path, mmap_mode='r')
  # normalize
  
  merging_rep = merging_rep_no_used/np.linalg.norm(merging_rep_no_used, axis=1, keepdims=True)
  decals_rep = decals_rep_no_used/np.linalg.norm(decals_rep_no_used, axis=1, keepdims=True)
  print(merging_rep.shape)
  print(decals_rep.shape)
  
  query_num = merging_rep.shape[0]

  file_dist = h5py.File(merging_decals_dist_path, 'w')
  dist_data = file_dist.create_dataset('dist', (query_num, decals_rep.shape[0]))


  for index in range(query_num):
    print(f"Processing {index}-th image")
    dist_data[index,:] = np.inner(merging_rep[index,:], decals_rep)
    # summary_chunk = np.maximum(dist_chunk, summary_chunk)

  file_dist.close()
  
def shrink_merging():
  merging_img = h5py.File(merging_img_path, 'r')
  file_output = h5py.File(merging_shrink_img_path, 'w')
  orig_size = 424
  new_size = 152

  print(merging_img['images'].shape)
  query_num = merging_img['images'].shape[0]
  result_data = file_output.create_dataset('images', (merging_img['images'].shape[0], 3, new_size, new_size))

  for index in range(query_num):
    for i in range(new_size):
      for j in range(new_size):
        result_data[index, :, i, j] = merging_img['images'][index, :, i*2, j*2]
      
  file_output.close()
  
def show_shrink_merging():
  merging_img = h5py.File(merging_shrink_img_path, 'r')
  decals_img = h5py.File(decals_img_path, 'r')
  
  print(decals_img['images'][0,:,:,:])
  
  nx = 4
  ny = 4
  f, axarr = plt.subplots(nx,ny)
  plt.subplots_adjust(wspace=0.01, hspace=0.01)
  last_one = 0
  for x in range(nx):
    for y in range(ny):
      axarr[x,y].imshow(np.transpose(merging_img['images'][last_one,:,:,:], (1, 2, 0)))
      axarr[x,y].axis('off')
      last_one += 1

  plt.axis('off')
  plt.show()
  merging_img.close()
  decals_img.close()
  

def calculate_dist_naive():
  merging_img = h5py.File(merging_shrink_img_path, 'r')
  decals_img = h5py.File(decals_img_path, 'r')
  file_dist = h5py.File(merging_decals_dist_naive_path, 'w')

  print(merging_img['images'].shape)
  print(decals_img['images'].shape)
  # query_num = merging_img['images'].shape[0]
  query_num = 20
  declas_num = decals_img['images'].shape[0]
  dist_data = file_dist.create_dataset('dist', (query_num, declas_num))
  
  # chunk_size = 100000
  # for index in range(query_num):
  #   print(f"Processing {index}-th image")
  #   chunk = declas_num // chunk_size
  #   if (declas_num % chunk_size) != 0:
  #     chunk += 1
  #   for i in range(chunk):
  #     start = i * chunk_size
  #     end = ((i+1) * chunk_size - 1)
  #     if (end >= declas_num):
  #       end = (declas_num - 1)
  #     dist = np.subtract(decals_img['images'][start:end,:,:,:], merging_img['images'][index,:,:,:])
  #     dist_data[index, start:end] = np.sum(np.multiply(dist, dist), axis = (1,2,3))
  
  # for index in range(query_num):
  #    print(f"Processing {index}-th image")
  #    for i in range(declas_num):
  #      dist = merging_img['images'][index,:,:,:] - decals_img['images'][i,:,:,:]
  #      dist_data[index, i] = math.sqrt(np.sum(np.multiply(dist, dist)))
  
  # chunk_size = 100000
  # for index in range(query_num):
  #   print(f"Processing {index}-th image")
  #   query_data = merging_img['images'][index,:,:,:]
  #   query_norm = np.sqrt(np.sum(np.multiply(query_data, query_data)))
  #   chunk = declas_num // chunk_size
  #   if (declas_num % chunk_size) != 0:
  #     chunk += 1
  #   for i in range(chunk):
  #     start = i * chunk_size
  #     end = ((i+1) * chunk_size - 1)
  #     if (end >= declas_num):
  #       end = (declas_num - 1)
  #     inner = np.sum(np.multiply(decals_img['images'][start:end,:,:,:], query_data), axis = (1,2,3))
  #     decals_norm = np.sqrt(np.sum(np.multiply(decals_img['images'][start:end,:,:,:], decals_img['images'][start:end,:,:,:]), axis = (1,2,3)))
  #     dist_data[index, start:end] = np.divide(inner / query_norm, decals_norm)
      

  for index in range(query_num):
    print(f"Processing {index}-th image")
    query_data = merging_img['images'][index,:,:,:].flatten()
    query_data = query_data/np.linalg.norm(query_data)
    for i in range(declas_num):
      decals_data = decals_img['images'][i,:,:,:].flatten()
      decals_data = decals_data/np.linalg.norm(decals_data)
      dist_data[index, i] = np.inner(query_data, decals_data)
      
  file_dist.close()

def result_galaxy_zoo():
  
  dist_data_no_used = h5py.File(galaxy_zoo_all_naive_dist_path, 'r')
  #dist_data_no_used = h5py.File(galaxy_zoo_all_dist_path, 'r')
  dist_data = dist_data_no_used['dist']
  
  sample_num = 700
  query_num = 700
  summary_data = np.zeros(dist_data.shape[1] - query_num, dtype=np.float32)

  #summary_data = np.amax(dist_data[0:699, query_num:], axis=0) 
  
  for index in range(sample_num):
    summary_data += dist_data[index, query_num:]

  summary_data /= (sample_num)
  print(f"similarity_score: {summary_data}")
  inds = np.argsort(summary_data, axis=0)[::-1]
  print(f"sorted index: {inds}")
  dist_top = np.take_along_axis(summary_data, inds, axis=0)
  print(f"sorted score: {dist_top}")
  
  ground_truth = 322
  
  for top in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]:
    correct_no = sum(count_list_by_condition(x, ground_truth) for x in inds[:top-1])
    precision = 100*correct_no / top
    recall = 100*correct_no / ground_truth
    print(f"top-{top}: precision={precision}, recall={recall}")

def count_list_by_condition(x, top_k):
  return x < top_k
  
def top_k():
  merging_img = h5py.File(merging_img_path, 'r')
  decals_img = h5py.File(decals_img_path, 'r')
  dist_data_no_used = h5py.File(merging_decals_dist_path, 'r')
  dist_data = dist_data_no_used['dist']
  
  #query_num = dist_data.shape[0]
  query_num = 1
  summary_data = np.zeros(dist_data.shape[1], dtype=np.float32)


  #print(merging_img['images'].shape)
  #print(decals_img['images'].shape)
  
  #tmp_index = np.argsort(dist_data[:,33866], axis=0)[::-1]
  #print(tmp_index)
  #tmp_top = np.take_along_axis(dist_data[:,33866], tmp_index, axis=0)
  #print(tmp_index[0:20])
  #print(tmp_top[0:20])
  
  #ignore_index = np.array([168, 510, 640])
  
  for index in range(query_num):
    #if (index in ignore_index): continue
    #summary_data = np.maximum(dist_data[index,:], summary_data)
    summary_data += dist_data[index,:]
  
  #query_index = 0
  #summary_data = dist_data[query_index,:]

  print(f"similarity_score: {summary_data}")
  inds = np.argsort(summary_data, axis=0)[::-1]
  print(f"sorted index: {inds}")
  dist_top = np.take_along_axis(summary_data, inds, axis=0)
  print(f"sorted score: {dist_top}")
  
  #top_indices = (dist_top>0.95)
  #kept_indices = np.extract(top_indices, inds)
  kept_indices = inds

  nx = 4
  ny = 4
  f, axarr = plt.subplots(nx,ny)
  plt.subplots_adjust(wspace=0.01, hspace=0.02)
  #last_one = kept_indices.shape[0] - 1
  last_one = 0
  print(f"There are total {kept_indices.shape[0]}")
  
  #print_index = [168, 510, 640, 478, 299, 57, 106, 351, 659, 981, 585, 301, 151, 716, 983, 54, 650, 394, 1016, 714]
  #count = 0
  

  
  #axarr[0,0].imshow(np.transpose(merging_img['images'][640,:,:,:], (1, 2, 0)))
  #axarr[0,0].imshow(np.transpose(decals_img['images'][33866,:,:,:], (1, 2, 0)))
  for x in range(nx):
    for y in range(ny):
      #if x==0 and y==0:
      #  continue
      axarr[x,y].imshow(np.transpose(decals_img['images'][kept_indices[last_one],:,:,:], (1, 2, 0)))
      #print(kept_indices[last_one])
      axarr[x,y].axis('off')
      last_one += 1

  plt.axis('off')
  plt.show()

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='runtime parameters')
  parser.add_argument("--calculate_dist", action="store_true",
                        help="calculate_dist")
  parser.add_argument("--calculate_dist_naive", action="store_true",
                        help="calculate_dist_naive")
  parser.add_argument("--calculate_top_k", action="store_true",
                        help="calculate_top_k")
  parser.add_argument("--shrink_merging", action="store_true",
                        help="shrink_merging")
  parser.add_argument("--show_shrink_merging", action="store_true",
                        help="show_shrink_merging")
  parser.add_argument("--calculate_dist_galaxy_zoo", action="store_true",
                        help="calculate_dist_galaxy_zoo")
  parser.add_argument("--result_galaxy_zoo", action="store_true",
                        help="result_galaxy_zoo")
  parser.add_argument("--calculate_galaxy_zoo_dist_naive", action="store_true",
                        help="calculate_galaxy_zoo_dist_naive")
  args = parser.parse_args()
    
  if args.calculate_dist:
    calculate_dist()
    
  if args.calculate_dist_naive:
    calculate_dist_naive()

  if args.shrink_merging:
    shrink_merging()
    
  if args.show_shrink_merging:
    show_shrink_merging()
  
  if args.calculate_top_k:
    top_k()
    
  if args.calculate_dist_galaxy_zoo:
    calculate_dist_galaxy_zoo()
    
  if args.result_galaxy_zoo:
    result_galaxy_zoo()
    
  if args.calculate_galaxy_zoo_dist_naive:
    calculate_galaxy_zoo_dist_naive()