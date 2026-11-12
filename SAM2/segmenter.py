# TEA-AgriVLN
# Segmenter

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tqdm import tqdm
import argparse
import sys
import yaml

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


np.random.seed(3)


def show_anns(anns, borders=True):
   if len(anns) == 0:
      return
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   ax = plt.gca()
   ax.set_autoscale_on(False)

   img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
   img[:, :, 3] = 0
   for ann in sorted_anns:
      m = ann['segmentation']
      color_mask = np.concatenate([np.random.random(3), [0.5]])
      img[m] = color_mask 
      if borders:
         import cv2
         contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
         cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

   ax.imshow(img)


def segmenter(name, image_path, output_folder_path):

   with open("../config.yaml", "r") as f:
      config = yaml.safe_load(f)
   size = config['segmenter']['size']

   # select the device for computation
   if torch.cuda.is_available():
      device = torch.device("cuda")
   elif torch.backends.mps.is_available():
      device = torch.device("mps")
   else:
      device = torch.device("cpu")

   if device.type == "cuda":
      torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
      if torch.cuda.get_device_properties(0).major >= 8:
         torch.backends.cuda.matmul.allow_tf32 = True
         torch.backends.cudnn.allow_tf32 = True
   
   image = np.array(Image.open(image_path).convert("RGB"))

   if size == 'base':
      sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
      model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
   elif size == 'large':
      sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
      model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

   sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

   mask_generator = SAM2AutomaticMaskGenerator(model=sam2,
      points_per_side=config['segmenter']['points_per_side'],
      pred_iou_thresh=config['segmenter']['pred_iou_thresh'],
      box_nms_thresh=config['segmenter']['box_nms_thresh'],
   )

   masks = mask_generator.generate(image)
   
   os.makedirs(output_folder_path, exist_ok=True)

   plt.figure(figsize=(20, 20))
   plt.imshow(image)
   show_anns(masks)
   plt.axis('off')
   plt.savefig(f"{output_folder_path}/{name}.png", bbox_inches='tight', dpi=300)
   plt.close()

   with open(f"{output_folder_path}/{name}.pkl", "wb") as f:
      pickle.dump(masks, f)


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--place", type=str, required=True, help="place")
   parser.add_argument("-i", "--id", type=int, required=True, help="id")
   args = parser.parse_args()
   place = args.place
   id = args.id

   print(f'[INFO] place: {place}, id: {id}')

   input_dir = f"../A2A/{place}_{id}/frames"
   frame_list = os.listdir(input_dir)
   frame_list = sorted(frame_list, key=lambda x: float(x.split("_")[-1].split(".")[0].replace("'", ".")))

   for frame in tqdm(frame_list, desc="segmenting"):
      
      name = os.path.splitext(frame)[0]
      input_path = f"{input_dir}/{frame}"
      output_dir = f"../after_segment/{place}_{id}"
      os.makedirs(output_dir, exist_ok=True)

      if os.path.isfile(f"{output_dir}/{name}.pkl") == True:
         print(f"{output_dir}/{name}.pkl exists.")
         continue

      segmenter(name, input_path, output_dir)

      print(f'{frame} success.')