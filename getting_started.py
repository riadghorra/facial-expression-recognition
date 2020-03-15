from dataset_tools import string_to_pilimage
from train import config
import numpy as np
import pandas as pd
import cv2

"""
Load Images from FER.
- config["path"] is the path to the dataset
- config["data_column"] is "pixels" (if the dataset is not cropped) or "face" (if dataset is cropped).
"""


def load_cv_images_from_fer(output_type, nrows=None):
    """
    :param type: "CV" for open cv and "PIL" for PIL
    :param nrows: nrows param for pd.read_csv
    :return: list of images from the fer csv file in "type" format.
    """
    imgs = []
    all_data = pd.read_csv(config["path"], header=0, nrows=nrows)
    for i in range(len(all_data)):
        pixels = all_data.loc[i][config["data_column"]]
        pil_img = string_to_pilimage(pixels)

        if output_type == "CV":
            imgs.append(np.array(pil_img))
        elif output_type == "PIL":
            imgs.append(pil_img)
        else:
            raise Exception("Invalid type")

    print("loaded images from csv")

    return imgs


def show_first_images():
    imgs = load_cv_images_from_fer("CV", nrows=3)

    for img in imgs:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        print(len(kp), des.shape)
        # img.show()


show_first_images()