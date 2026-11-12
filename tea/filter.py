# TEA-AgriVLN
# Filter

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml


def calculate_iou(mask1, mask2):
   
   seg1 = mask1["segmentation"]
   seg2 = mask2["segmentation"]

   intersection = np.logical_and(seg1, seg2).sum()
   union = np.logical_or(seg1, seg2).sum()

   if union == 0:
      return 0.0

   return intersection / union


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
         # Try to smooth contours
         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
         cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

   ax.imshow(img)


def filter(name, image_path, input_dir, output_dir, full_area):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
   iou_threshold = config['filter']['iou_threshold']
   threshold_rate = config['filter']['threshold_rate']
   
   threshold = threshold_rate * full_area

   pkl_path = os.path.join(input_dir, f"{name}.pkl")
   with open(pkl_path, "rb") as f:
      before_mask_list = pickle.load(f)

   after_mask_list = []

   # area
   for before_mask in before_mask_list:
      if before_mask['area'] > threshold:
         after_mask_list.append(before_mask)

   # sort
   after_mask_list = sorted(
      after_mask_list,
      key=lambda x: x["area"],
      reverse=True
   )

   # iou clean
   final_mask_list = []
   
   for i, mask in enumerate(after_mask_list):
      keep = True

      for kept_mask in final_mask_list:
         iou = calculate_iou(mask, kept_mask)

         if iou > iou_threshold:
            keep = False
            break

      if keep:
         final_mask_list.append(mask)
   
   after_mask_list = final_mask_list

   with open(os.path.join(output_dir, f"{name}.pkl"), "wb") as f:
      pickle.dump(after_mask_list, f)

   image = np.array(Image.open(image_path).convert("RGB"))
   plt.figure(figsize=(20, 20))
   plt.imshow(image)
   show_anns(after_mask_list)
   plt.axis('off')
   plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', dpi=300)
   plt.close()
   