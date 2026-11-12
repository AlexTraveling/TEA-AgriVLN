# TEA-AgriVLN
# Overlay

import cv2
import pickle
import numpy as np
import os


def overlay(
   image_path,
   pkl,
   output_path,
   alpha=0.5,
   color=(245, 245, 245),
   border=True,
   border_color = (147, 20, 255)
):
   
   image = cv2.imread(image_path)

   segmentation = pkl["segmentation"]

   # Create overlay
   overlay = image.copy()

   # Fill the mask region
   overlay[segmentation] = color

   # Blend only inside the mask
   result = image.copy()
   result[segmentation] = (
      alpha * overlay[segmentation]
      + (1 - alpha) * image[segmentation]
   ).astype(np.uint8)

   # Draw contour if needed
   if border:
      contours, _ = cv2.findContours(
         segmentation.astype(np.uint8),
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
         border_color,
         thickness=2,
      )

   cv2.imwrite(output_path, result)
   