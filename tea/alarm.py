# TEA-AgriVLN
# Alarm

import cv2
import numpy as np
import pickle
import yaml
import sys
import os
import matplotlib.pyplot as plt


def show(image_path, output_path):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)

   top_ratio = config['alarm']['top_ratio']
   bottom_ratio = config['alarm']['bottom_ratio']
   height_ratio = config['alarm']['height_ratio']

   image = cv2.imread(image_path)
   H, W = image.shape[:2]

   top_w = int(top_ratio * W)
   bottom_w = int(bottom_ratio * W)
   h = int(height_ratio * H)

   y_bottom = H
   y_top = H - h

   x_top_left = (W - top_w) // 2
   x_top_right = (W + top_w) // 2
   x_bottom_left = (W - bottom_w) // 2
   x_bottom_right = (W + bottom_w) // 2

   pts = np.array([
      [x_top_left, y_top],
      [x_top_right, y_top],
      [x_bottom_right, y_bottom],
      [x_bottom_left, y_bottom],
   ], dtype=np.int32)

   vis = image.copy()

   cv2.polylines(
      vis,
      [pts],
      True,
      (255, 255, 255),
      2,
      cv2.LINE_AA
   )

   cv2.imwrite(output_path, vis)


def calculate(pkl_list_path):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
   camera_height = config['camera']['height']
   camera_width = config['camera']['width']
   top_ratio = config['alarm']['top_ratio']
   bottom_ratio = config['alarm']['bottom_ratio']
   height_ratio = config['alarm']['height_ratio']

   # get ladder
   ladder = np.zeros((camera_height, camera_width), dtype=bool)

   trap_height = int(camera_height * height_ratio)
   top_width = camera_width * top_ratio
   bottom_width = camera_width * bottom_ratio
   center_x = camera_width / 2

   for i in range(trap_height):
      y = camera_height - 1 - i
      ratio = i / max(trap_height - 1, 1)
      current_width = bottom_width * (1 - ratio) + top_width * ratio
      left = int(center_x - current_width / 2)
      right = int(center_x + current_width / 2)
      left = max(left, 0)
      right = min(right, camera_width)
      ladder[y, left:right] = True

   # segment go on
   with open(pkl_list_path, "rb") as f:
      pkl_list = pickle.load(f)

   # split
   t_pkl_list = []
   n_pkl_list = []
   for pkl in pkl_list:
      if pkl['traversability'] == 'yes':
         t_pkl_list.append(pkl)
      if pkl['traversability'] == 'no':
         n_pkl_list.append(pkl)

   # get segment
   t_segment_list = []
   for t_pkl in t_pkl_list:
      t_segment_list.append(t_pkl['segmentation'])
   n_segment_list = []
   for n_pkl in n_pkl_list:
      n_segment_list.append(n_pkl['segmentation'])

   # get full segment
   t_segment_full = np.any(t_segment_list, axis=0)
   n_segment_full = np.any(n_segment_list, axis=0)

   # get overlap
   t_inside = np.logical_and(ladder, t_segment_full)
   n_inside = np.logical_and(ladder, n_segment_full)

   t_sum = np.sum(t_inside)
   n_sum = np.sum(n_inside)
   ladder_sum = np.sum(ladder)

   t_ratio = round(t_sum / ladder_sum, 3)
   n_ratio = round(n_sum / ladder_sum, 3)

   return t_ratio, n_ratio
   

def judge(action, t_ratio, n_ratio):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
   traversable_threshold = config['alarm']['traversable_threshold']
   non_traversable_threshold = config['alarm']['non_traversable_threshold']

   if action == '[FORWARD]':
      if n_ratio > non_traversable_threshold:
         return 'on'
      else:
         return 'off'
   
   if action == '[STOP]':
      if n_ratio < non_traversable_threshold and t_ratio > traversable_threshold:
         return 'on'
      else:
         return 'off'
   
   sys.exit('[ERROR] The action is not available.')
      

def alarm(action, episode, name):

   # debug 2027.7.13
   if action == '[LEFT ROTATE]' or action == '[RIGHT ROTATE]':
      return 'off'

   # show
   image_path = f"after_classifier/{episode}/{name}.png"
   output_dir = f"after_alarm/{episode}"
   os.makedirs(output_dir, exist_ok=True)
   output_path = f"{output_dir}/{name}.png"

   # debug 2027.7.13
   if os.path.exists(image_path) == False:
      return 'off'
   
   show(image_path, output_path)

   # calculate area
   pkl_list_path = f"after_classifier/{episode}/{name}.pkl"
   t_ratio, n_ratio = calculate(pkl_list_path)

   # alarm
   a = judge(action, t_ratio, n_ratio)

   return a
