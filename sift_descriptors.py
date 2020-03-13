import cv2
from getting_started import load_cv_images_from_fer
import numpy as np

"""
Must use specific opencv version for sift to work : 
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
"""

imgs = load_cv_images_from_fer("CV", nrows=3)

img = imgs[0]
sift = cv2.xfeatures2d.SIFT_create()
keypoints_sift, descriptors = sift.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints_sift, None)


cv2.imwrite('sift_keypoints.jpg', img)
