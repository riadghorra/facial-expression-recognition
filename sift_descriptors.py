import cv2

from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt

from getting_started import load_cv_images_from_fer

"""
Must use specific opencv version for sift to work : 
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
"""


class DenseDetector:
    def __init__(self, step_size=12, feature_scale=12, img_bound=6):
        self.step = step_size
        self.feature_scale = feature_scale
        self.img_bound = img_bound
        self.detector = cv2.xfeatures2d.SIFT_create()

    def detect_keypoints(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.img_bound, rows, self.feature_scale):
            for y in range(self.img_bound, cols, self.feature_scale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.step))
        return keypoints

    def compute_descriptors(self, img):
        kp = self.detect_keypoints(img)
        try:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            gray_image = img
        return self.detector.compute(gray_image, kp)


class SIFTDetector:
    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()

    def detect_keypoints(self, img):
        try:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            gray_image = img
        return self.detector.detect(gray_image, None)

    def compute_descriptors(self, img):
        try:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            gray_image = img
        return self.detector.detectAndCompute(gray_image, None)


def bulk_extract_features(images, detector):
    descriptors = []
    for im in tqdm(images):
        descriptors.append(detector.compute_descriptors(im)[1])

    return descriptors


def display_image_with_keypoints(image, detector):
    keypoints = detector.detect_keypoints(image)
    im_with_kp = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(im_with_kp, interpolation='nearest')
    plt.show()


def apply_detector_to_dataset():
    input_images = load_cv_images_from_fer(output_type='CV', nrows=None)
    dense_detector = DenseDetector()
    sift_detector = SIFTDetector()

    dense_descriptors = bulk_extract_features(input_images, dense_detector)
    sift_descriptors = bulk_extract_features(input_images, sift_detector)

    with open('sift_descriptors/ferplus_dense_descriptors.pkl', 'wb') as f:
        pickle.dump(dense_descriptors, f)

    with open('sift_descriptors/ferplus_sift_descriptors.pkl', 'wb') as f:
        pickle.dump(sift_descriptors, f)

# apply_detector_to_dataset()