import pickle
import numpy as np
import pandas as pd


def flatten_desc(desc):
    """
    Transform list of descriptors into numpy array of size (number_of_images, number_of_features)
    :param desc: list of descriptors extracted from images in the form of arrays of size (16,128)
    :return: numpy array of size (number_of_images, number_of_keypoints * number_of_descriptors)
    """
    l = [x for x in desc if x is not None]
    l = [x.flatten() for x in l]
    return np.vstack(l)


def numpy_to_string(arr):
    """
    Converts numpy array of dimension 1 to string with values separated by a ' '
    :param arr: numpy array of dimension 1 or list
    :return: string
    """
    return ' '.join([str(x) for x in arr])


def bulk_convert_matrix_to_string(mat):
    """
    Converts matrix or numpy array of dimension 2 to list of pixels
    :param mat: numpy array of dimension 2
    :return: list of strings
    """

    list_of_strings = []
    for arr in mat:
        list_of_strings.append(numpy_to_string(arr))

    return list_of_strings


def build_csv():
    with open('sift_descriptors/ferplus_dense_descriptors.pkl', 'rb') as f:
        fdd = pickle.load(f)

    with open('sift_descriptors/ferpluscropped_dense_descriptors.pkl', 'rb') as f:
        fcdd = pickle.load(f)

    ferplus = pd.read_csv("fer_datasets/ferplus.csv")
    ferplus_cropped = pd.read_csv("fer_datasets/ferplus_cropped.csv")

    fdd_arr = flatten_desc(fdd)
    fcdd_arr = flatten_desc(fcdd)

    ferplus["descriptors"] = bulk_convert_matrix_to_string(fdd_arr)
    ferplus_cropped["descriptors"] = bulk_convert_matrix_to_string(fcdd_arr)

    ferplus.to_csv("sift_descriptors/ferplus_sift_labels.csv", index=False)
    ferplus_cropped.to_csv("sift_descriptors/ferplus_cropped_sift_labels.csv", index=False)

build_csv()

"""
There are no NaN in either these dataframes
"""
