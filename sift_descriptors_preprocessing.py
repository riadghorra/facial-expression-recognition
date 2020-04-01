import pickle
import numpy as np
import pandas as pd


def extract_labels_from_ferplus(df):
    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    df['Max'] = df[labels].idxmax(axis=1)
    return df["Max"].map(labels.index)




def flatten_desc(desc):
    """
    Transform list of descriptors into numpy array of size (number_of_images, number_of_features)
    :param desc: list of descriptors extracted from images in the form of arrays of size (16,128)
    :return: numpy array of size (number_of_images, number_of_keypoints * number_of_descriptors)
    """
    l = [x for x in desc if x is not None]
    l = [x.flatten() for x in l]
    return np.vstack(l)


def add_labels(arr, lab):
    """
    Adds classification labels on the descriptors array

    :param arr: flattened descriptors array
    :return: numpy array with labels
    """
    df = pd.DataFrame(arr, columns=["var_{}".format(i) for i in range(len(arr[0]))])
    df["label"] = lab
    return df


def build_csv():
    with open('sift_descriptors/ferplus_dense_descriptors.pkl', 'rb') as f:
        fdd = pickle.load(f)

    with open('sift_descriptors/ferpluscropped_dense_descriptors.pkl', 'rb') as f:
        fcdd = pickle.load(f)

    ferplus = pd.read_csv("fer_datasets/ferplus.csv")
    ferplus_cropped = pd.read_csv("fer_datasets/ferplus_cropped.csv")
    labels_cropped = ferplus_cropped.emotion.tolist()
    labels = extract_labels_from_ferplus(ferplus)

    df = add_labels(flatten_desc(fdd), labels)
    df.to_csv("ferplus_sift_labels.csv", index=False)
    df1 = add_labels(flatten_desc(fcdd), labels_cropped)
    df1.to_csv("ferplus_cropped_sift_labels.csv", index=False)

"""
There are no NaN in either these dataframes
"""
