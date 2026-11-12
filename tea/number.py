# TEA-AgriVLN
# Number

import cv2
import os
import pickle
import sys


def number(image_path, value, output_dir, output_path):
   image = cv2.imread(image_path)

   text = f"segment #{value}"
   font = cv2.FONT_HERSHEY_SIMPLEX
   font_scale = 1.0
   thickness = 2
   margin = 10

   (text_w, text_h), baseline = cv2.getTextSize(
      text, font, font_scale, thickness
   )

   overlay = image.copy()

   x1 = margin
   y1 = margin
   x2 = x1 + text_w + 20
   y2 = y1 + text_h + baseline + 20

   cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

   alpha = 0.5
   image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

   text_x = x1 + 10
   text_y = y1 + text_h + 10
   cv2.putText(
      image,
      text,
      (text_x, text_y),
      font,
      font_scale,
      (255, 255, 255),
      thickness,
      cv2.LINE_AA
   )

   os.makedirs(output_dir, exist_ok=True)

   cv2.imwrite(output_path, image)
