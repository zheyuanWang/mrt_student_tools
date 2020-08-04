
import itertools
import os.path as osp
import argparse
import cv2
from urllib.request import urlretrieve
import numpy as np
from tqdm import tqdm
import time
import os
from multiprocessing import cpu_count
from multiprocessing import Pool

from model.resunet import ResUNetBN2D2
from util.visualization import visualize_image_correspondence

import torch

import matplotlib.pyplot as plt
from util.file import ensure_dir

if not osp.isfile('ResUNetBN2D2-YFCC100train.pth'):
  print('Downloading weights...')
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/ucn/ResUNetBN2D2-YFCC100train-100epoch.pth",
      'ResUNetBN2D2-YFCC100train.pth')

imgs = [
    '00193173_7195353638.jpg',
    '01058134_62294335.jpg',
    '01462567_5517704156.jpg',
    '01712771_5951658395.jpg',
    '02097228_5107530228.jpg',
    '04240457_5644708528.jpg',
    '04699926_7516162558.jpg',
    '05140127_5382246386.jpg',
    '05241723_5891594881.jpg',
    '06903912_8664514294.jpg',
]
imgs_carla = [
    'img_l.jpg',
    'img_r.jpg',
]


def prep_image(full_path):
  assert osp.exists(full_path), f"File {full_path} does not exist."
  return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)


def to_normalized_torch(img, device):
  """
  Normalize the image to [-0.5, 0.5] range and augment batch and channel dimensions.
  """
  img = img.astype(np.float32) / 255 - 0.5
  return torch.from_numpy(img).to(device)[None, None, :, :]


def demo(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  root = './imgs_carla'
  checkpoint = torch.load(config.weights)
  model = ResUNetBN2D2(1, 64, normalize_feature=True)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  model = model.to(device)

  # Try all combinations
  # f_l = cv2.imread('/home/suyang/workspaces/catkin_ws/src/open-ucn/imgs_f/f_l.png', 0)
  # f_r = cv2.imread('/home/suyang/workspaces/catkin_ws/src/open-ucn/imgs_f/f_r.png', 0)
  # f_l = np.reshape(f_l, (1, 512, 512))
  # f_r = np.reshape(f_r, (1, 512, 512))
  # f_l = torch.from_numpy(f_l)
  # f_r = torch.from_numpy(f_r)

  for num_epi in range(1):
    path = "/mrtstorage/users/suyang/gnnet_benchmark_v1.0/GNNET_BENCHMARK_PUBLIC/carla_training_validation/all_weathers_eval/episode_"
    path_img = []
    path_f = []
    path_dir = []
    list = []
    for num_image in tqdm(range(0, 500)):
        start = time.time()
        # /mrtstorage/users/suyang/gnnet_benchmark_v1.0/GNNET_BENCHMARK_PUBLIC/carla_training_validation/all_weathers/episode_000/CameraRGB0
        path_img_l = path + ("%03d" % num_epi) + "/CameraRGB0/image_" + ("%05d" % num_image) + ".png"
        # path_img_r = path + ("%03d" % num_epi) + "/CameraRGB1/image_" + ("%05d" % num_image) + ".png"
        path_f_l = path + ("%03d" % num_epi) + "/feature_l_new_norm/image_" + ("%05d" % num_image) + ".png"
        # path_f_r = path + ("%03d" % num_epi) + "/feature_r/image_" + ("%05d" % num_image) + ".png"
        path_dir_l = path + ("%03d" % num_epi) + "/feature_l_new_norm"
        # path_dir_r = path + ("%03d" % num_epi) + "/feature_r"

        # path_img.append(path_img_l)
        # path_f.append(path_f_l)
        # path_dir.append(path_dir_l)
        list.append([path_dir_l, path_img_l, path_f_l])

    def func(list):
        _path_dir = list[0]
        _path_img = list[1]
        _path_f = list[2]
        folder = os.path.exists(_path_dir)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(_path_dir)
        img = prep_image(_path_img)
        F = model(to_normalized_torch(img, device))
        f = F[0][0].cpu().numpy()
        cv2.normalize(f, f, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(_path_f, f)


    pool = Pool(processes=6)
    pool.map(func, list)
    pool.close()
    pool.join()

    end = time.time()
    print("time: ", end - start)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--weights',
      default='best_val_checkpoint5.pth',
      type=str,
      help='Path to pretrained weights')
  parser.add_argument(
      '--nn_max_n',
      default=25,
      type=int,
      help='Number of maximum points for nearest neighbor search.')
  parser.add_argument(
      '--ucn_inlier_threshold_pixel',
      default=4,
      type=int,
      help='Max pixel distance for reciprocity test.')

  config = parser.parse_args()

  with torch.no_grad():
    demo(config)
