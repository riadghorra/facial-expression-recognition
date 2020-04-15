import numpy as np
import pandas as pd
import PIL as pl
import json

with open('config.json') as json_file:
    config = json.load(json_file)

"""
Load Images from FER.
- config["path"] is the path to the dataset
- config["data_column"] is "pixels" (if the dataset is not cropped) or "face" (if dataset is cropped).
"""


def pixelstring_to_numpy(string, flatten=False, integer_pixels=False):
    pixels = string.split()
    if flatten:
        out = np.array([int(i) for i in pixels])
        return out
    out = np.zeros((48, 48))
    for i in range(48):
        out[i] = np.array([int(k) for k in pixels[48 * i:48 * (i + 1)]])

    if integer_pixels:
        return out

    return out / 255.0


def string_to_pilimage(pixelstring):
    imarray = pixelstring_to_numpy(pixelstring, integer_pixels=True)
    out = pl.Image.fromarray(imarray).convert("L")
    return out


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
    imgs = load_cv_images_from_fer("PIL", nrows=3)

    for img in imgs:
        img.show()

