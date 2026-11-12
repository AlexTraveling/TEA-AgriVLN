# TEA-AgriVLN
# TEA Homepage

import argparse
import cv2
import numpy as np
import pickle
import sys
import os
import yaml
from tqdm import tqdm

from tea.filter import filter
from tea.overlay import overlay
from tea.number import number
from tea.classifier import classifier


def visualize(
   image_path,
   pkl_list_1,
   pkl_list_2,
   output_path,
   alpha=0.5,
   color_1=(185, 225, 255),   # Traversable (Khaki, BGR)
   color_2=(210, 210, 255),   # Non-traversable (Sakura Pink, BGR)
   border=True,
):
   # Read image
   image = cv2.imread(image_path)
   result = image.copy()

   # Draw one mask list
   def draw_masks(mask_list, color):
      nonlocal result

      overlay = result.copy()

      for pkl in mask_list:
         segmentation = pkl["segmentation"]

         # Fill mask
         overlay[segmentation] = color

         # Alpha blending
         result[segmentation] = (
               alpha * overlay[segmentation]
               + (1 - alpha) * result[segmentation]
         ).astype("uint8")

         # Border
         if border:
               contours, _ = cv2.findContours(
                  segmentation.astype("uint8"),
                  cv2.RETR_EXTERNAL,
                  cv2.CHAIN_APPROX_NONE,
               )
               contours = [
                  cv2.approxPolyDP(contour, epsilon=1.0, closed=True)
                  for contour in contours
               ]
               cv2.drawContours(
                  result,
                  contours,
                  -1,
                  color,
                  thickness=2,
               )

   draw_masks(pkl_list_1, color_1)  # Traversable
   draw_masks(pkl_list_2, color_2)  # Non-traversable

   cv2.imwrite(output_path, result)


if __name__ == '__main__':

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
   full_area = config['camera']['width'] * config['camera']['height']
   color = tuple(config['overlay']['color'])

   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--place", type=str, required=True, help="place")
   parser.add_argument("-i", "--id", type=int, required=True, help="id")
   args = parser.parse_args()
   place = args.place
   id = args.id

   input_dir = f'after_segment/{place}_{id}'
   output_dir = f'after_filter/{place}_{id}'
   os.makedirs(output_dir, exist_ok=True)

   frame_list = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
   frame_list = sorted(frame_list, key=lambda x: float(x.split("_")[-1].split(".")[0].replace("'", ".")))
   name_list = [os.path.splitext(f)[0] for f in frame_list]
   
   # filter
   if_filter = config['home_tea']['if_filter']
   if if_filter == True:
      for name in tqdm(name_list, desc="filtering"):
         image_path = f'A2A/{place}_{id}/frames/{name}.jpg'
         filter(name, image_path, input_dir, output_dir, full_area)
      print('[INFO] filter success.')
   else:
      print('[INFO] skip filter.')
   
   # overlay
   if_overlay = config['home_tea']['if_overlay']
   if if_overlay == True:
      input_dir = f'after_filter/{place}_{id}'
      output_dir = f'after_overlay/{place}_{id}'
      os.makedirs(output_dir, exist_ok=True)

      frame_list = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
      frame_list = sorted(frame_list, key=lambda x: float(x.split("_")[-1].split(".")[0].replace("'", ".")))
      name_list = [os.path.splitext(f)[0] for f in frame_list]

      for name in tqdm(name_list, desc="overlaying"):

         # read segmentation
         pkl_path = os.path.join(input_dir, f"{name}.pkl")
         with open(pkl_path, "rb") as f:
            pkl_list = pickle.load(f)
         for n in range(len(pkl_list)):
            pkl_list[n]["number"] = n + 1

         # what about 0?
         if len(pkl_list) == 0:
            print(f'[INFO] {name} is 0!')
            continue

         # save pkl
         with open(os.path.join(output_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(pkl_list, f)
         image_path = f"A2A/{place}_{id}/frames/{name}.jpg"
         for pkl in pkl_list:
            value = pkl['number']
            output_path = f"{output_dir}/{name}_{value}.png"
            overlay(image_path, pkl, output_path=output_path, color=color, border_color=color)
      print('[INFO] overlaying success.')
   else:
      print('[INFO] skip overlaying.')
   
   # number
   if_number = config['home_tea']['if_number']
   
   input_dir = f'after_overlay/{place}_{id}'
   output_dir = f'after_number/{place}_{id}'
   os.makedirs(output_dir, exist_ok=True)

   frame_list = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
   frame_list = sorted(frame_list, key=lambda x: float(x.split("_")[-1].split(".")[0].replace("'", ".")))
   name_list = [os.path.splitext(f)[0] for f in frame_list]
   
   if if_number == True:
      
      for name in tqdm(name_list, desc="numbering"):

         # read segmentation
         pkl_path = os.path.join(input_dir, f"{name}.pkl")
         with open(pkl_path, "rb") as f:
            pkl_list = pickle.load(f)

         for pkl in pkl_list:
            value = pkl['number']
            input_path = f"{input_dir}/{name}_{value}.png"
            output_path = f"{output_dir}/{name}_{value}.png"
            number(input_path, value, output_dir, output_path)
      print('[INFO] numbering success.')
   else:
      print('[INFO] skip numbering.')

   # classifier
   if_classifier = config['home_tea']['if_classifier']
   if if_classifier == True:

      with open("config.yaml", "r") as f:
         config = yaml.safe_load(f)

      limit = 0
      limit_max = 999
      for name in tqdm(name_list, desc="classifying"):

         # get segment quantity
         pkl_path = f'after_overlay/{place}_{id}/{name}.pkl'
         with open(pkl_path, "rb") as f:
            pkl_list = pickle.load(f)
         segment_quantity = len(pkl_list)

         # l_0: original image
         image_path_list = [f"A2A/{place}_{id}/frames/{name}.jpg"]

         for i in range(1, segment_quantity + 1):
            # l_n: marked segment
            image_path_list.append(f"after_number/{place}_{id}/{name}_{i}.png")

         output_dir = f'after_classifier/{place}_{id}'
         os.makedirs(output_dir, exist_ok=True)

         classifier(place, id, name, image_path_list, output_dir, if_VLM=True)

         limit += 1
         if limit > limit_max:
            break
         
      print('[INFO] classifying success.')
   else:
      print('[INFO] skip classifying.')

   # visualize
   for name in tqdm(name_list, desc="visualizing"):

      image_path = f"A2A/{place}_{id}/frames/{name}.jpg"

      pkl_list_path = f"after_classifier/{place}_{id}/{name}.pkl"
      with open(pkl_list_path, "rb") as f:
         pkl_list = pickle.load(f)

      # traversable
      pkl_list_t = [pkl for pkl in pkl_list if pkl["traversability"] == "yes"]
      # non-traversable
      pkl_list_n = [pkl for pkl in pkl_list if pkl["traversability"] == "no"]

      output_path = f"after_classifier/{place}_{id}/{name}.png"

      visualize(image_path, pkl_list_t, pkl_list_n, output_path)

   print('[INFO] visualizing success.')
   print(f'[INFO] {place}_{id} home_tea.py success.')