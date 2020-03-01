import PIL as pl
import torch
import torchvision.transforms as transforms
from time import time
from classifier import Custom_vgg
import os
import pandas as pd
import numpy as np

from dataset_tools import preprocess_batch_custom_vgg
from pipeline import crop_faces, crop_cv_img
from train import evaluate, DEVICE, config
from utils import plot_confusion_matrix

pre_process = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])


def resize_input_image(pil_img):
    (w, h) = pil_img.size
    aspect_ratio = w / h

    if w > 800:
        return pil_img.resize((800, int(800 / aspect_ratio)))

    if h > 800:
        return pil_img.resize((int(800 * aspect_ratio), 800))

    return pil_img


def load_model():
    model = Custom_vgg(1, len(config["catslist"]), DEVICE)
    model.load_state_dict(torch.load(config["current_best_model"], map_location=DEVICE))
    return model


def test_on_fer_test_set(fer_path):
    start_time = time()
    fer = pd.read_csv(fer_path)
    if "attribution" not in fer:
        raise Exception("Fer not split between train/val/test. Please run split_fer script.")
    fer_test = fer[fer["attribution"] == "test"]

    model = load_model()

    print("Loaded fer test set and model in {}s".format(time() - start_time))
    start_time = time()

    def preprocess_batch(pixelstring_batch, emotions_batch, DEVICE):
        return preprocess_batch_custom_vgg(pixelstring_batch, emotions_batch, DEVICE, False, config["loss_mode"])

    dummy_weights = torch.FloatTensor([1]*len(config["catslist"])).to(DEVICE)  # we don't care about the test loss value here.
    proba, _, acc, cm1, cm2, acc_fact = evaluate(model, fer_test, preprocess_batch, dummy_weights, DEVICE, compute_cm=True)

    print("FINAL ACCURACY: {}".format(acc))
    print("Average predicted proba for right class: {}".format(proba))
    print("Duration on {} test faces: {}s".format(len(fer_test), round(time() - start_time, 2)))
    print("Accuracy with grouped classes : {}".format(acc_fact))
    print("Close the confusion matrices to end the script.")
    plot_confusion_matrix(cm1, config["catslist"])
    plot_confusion_matrix(cm2, ["bad", "good", "surprise", "neutral"])


def test_on_folder():
    images = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(config["path_images"]):
        for file in f:
            if any(extension in file for extension in ['.jpeg', '.jpg', '.png']):
                images.append(os.path.join(r, file))

    results_df = pd.DataFrame()

    IMG_TO_TEST = 200
    i = 0

    for path in images:
        if i >= IMG_TO_TEST:
            break
        i += 1
        print("Processing {}".format(path))
        image = pl.Image.open(path)
        image = resize_input_image(image)
        cv_img = np.array(image)
        [face_coords] = crop_faces([cv_img])
        if face_coords is not None:
            print("Face found on image.")
            (x, y, w, h) = face_coords
            face_image = crop_cv_img(cv_img, x, y, w, h)
            pil_face_image = pl.Image.fromarray(face_image).resize(config["resolution"])
            pil_face_image = pre_process(pil_face_image).unsqueeze(0)

            model = load_model()
            results = model.predict_single(pil_face_image)
            results_dict = {cat: results[i] for i, cat in enumerate(config["catslist"])}
            results_dict["path"] = path
            results_df = results_df.append(results_dict, ignore_index=True)

    results_df.to_csv("predictions.csv", index=False)


def test_on_annotated_csv(annotations_csv_path):
    start_time = time()
    # add column "pixels" to annotated csv (same format as FER csv)
    print("Loading annotations...")
    df = pd.read_csv(annotations_csv_path)
    pixels = []
    for path in df["path"].values:
        image = np.array(pl.Image.open(path))
        pixels_list = image.flatten().tolist()
        pixels.append(" ".join(map(str, pixels_list)))
    df[config["data_column"]] = pixels

    print("Loaded annotations in {}s".format(round(time() - start_time, 2)))
    start_time = time()
    # load model and evaluate it
    print("Loading model...")
    model = load_model()

    print("Evaluating model...")
    
    def preprocess_batch(pixelstring_batch, emotions_batch, DEVICE):
        return preprocess_batch_custom_vgg(pixelstring_batch, emotions_batch, DEVICE, False, config["loss_mode"])

    dummy_weights = torch.FloatTensor([1]*len(config["catslist"])).to(DEVICE)  # we don't care about the test loss value here.
    proba, _, acc, cm1, cm2, acc_fact = evaluate(model, df, preprocess_batch, dummy_weights, DEVICE, compute_cm=True)

    print("FINAL ACCURACY: {}".format(acc))
    print("Average predicted proba for right class: {}".format(proba))
    print("Duration on {} test faces: {}s".format(len(df), round(time() - start_time, 2)))
    print("Close the confusion matrix to end the script.")
    print("Accuracy with grouped classes : {}".format(acc_fact))
    plot_confusion_matrix(cm1, config["catslist"])
    plot_confusion_matrix(cm2, ["bad", "good", "surprise", "neutral"])


test_on_annotated_csv("./annotations.csv")
