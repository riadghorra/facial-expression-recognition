import cv2

import numpy as np
from matplotlib import pyplot as plt

from getting_started import load_cv_images_from_fer

"""
Must use specific opencv version for sift to work : 
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
"""

class DenseDetector(): 
    def __init__(self, step_size=20, feature_scale=20, img_bound=20): 
        self.step = step_size
        self.feature_scale = feature_scale
        self.img_bound = img_bound
 
    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.img_bound, rows, self.feature_scale):
            for y in range(self.img_bound, cols, self.feature_scale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.step))
        return keypoints 

class SIFTDetector():
    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()

    def detect(self, img):
        try:
          gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        except:
          gray_image = img
        return self.detector.detect(gray_image, None)  

if __name__=='__main__':
    input_image = load_cv_images_from_fer(output_type='CV', nrows=1)[0]
    input_image_dense = np.copy(input_image)
    input_image_sift = np.copy(input_image)

    keypoints = DenseDetector(20,20,5).detect(input_image)
    input_image_dense = cv2.drawKeypoints(input_image_dense, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

    plt.imshow(input_image_dense,interpolation='nearest')
    plt.show()

    keypoints = SIFTDetector().detect(input_image)
    input_image_sift = cv2.drawKeypoints(input_image_sift, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

    plt.imshow(input_image_sift,interpolation='nearest')
    plt.show()