import torch

import cv2
import os
import json
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import transforms

from classifier import Custom_vgg
from dataset_tools import string_to_pilimage

with open('config.json') as json_file:
    config = json.load(json_file)


def get_img_path():
    """
    :return: list of all paths for images contained in the folder config["path_images"]:
    """
    paths = []
    for r, d, f in os.walk(config["path_images"]):
        for file in f:
            if any(extension in file for extension in ['.jpeg', '.jpg', '.png']):
                paths.append(os.path.join(r, file))
    return paths


def load_cv_imgs(paths):
    """
    :return: load opencv images from list of image paths.
    """
    imgs = []
    for index, path in enumerate(paths):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgs.append(gray_img)

    return imgs


def load_cv_images_from_fer():
    """
    :return: list of opencv images from the fer csv file.
    """
    cv_imgs = []
    all_data = pd.read_csv(config["path"], header=0, nrows=30000)
    for i in range(30000):
        pixels = all_data.loc[i]['pixels']
        pil_img = string_to_pilimage(pixels)
        cv_imgs.append(np.array(pil_img))

    print("loaded images from csv")

    return cv_imgs


def crop_faces(cv_imgs):
    """
    Return an array of coordinates of faces, one face per image.
    If none or several faces were found on an image, the coordinates for this image are None.

        Tested parameters for the face recognition with opencv on the FER dataset :
        Params:
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        Results:
            No face:  14257
            One face:  15733
            Several faces:  10

        Params:
            scaleFactor=1.01,
            minNeighbors=3,
            minSize=(20, 20)
        Results:
            No face:  9601
            One face:  20242
            Several faces:  157

    :param cv_imgs: opencv image array.
    :return: array of coordinates of the face for each input image (which can contain be None values).
    """
    faces_coords = []
    img_with_several_faces = 0
    img_with_no_face = 0
    img_with_one_face = 0
    for index, image in enumerate(cv_imgs):
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.01,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 1:
            img_with_one_face += 1
            (x, y, w, h) = faces[0]
            faces_coords.append((x, y, w, h))
        else:
            faces_coords.append(None)
            if len(faces) == 0:
                img_with_no_face += 1
            else:
                img_with_several_faces += 1

    return faces_coords


def make_video(fps):
    """
    Make video from model predictions on the youtube faciale expression video: https://www.youtube.com/watch?v=B0ouAnmsO1Y
     - Take one frame over 3 on the original 60fps video
     - Detect face on the frame and draw rectangle delimiter.
     - If a face is detected, run the model on it and write the results on the frame.
    :param fps:
    :return:
    """
    cap = cv2.VideoCapture('./videos/facial_expressions_demo.mov')
    out = cv2.VideoWriter(
        './videos/output.avi',
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fps,
        (int(cap.get(3)), int(cap.get(4)))
    )

    model = Custom_vgg(1, config["cats"], device=torch.device('cpu'))
    model.load_state_dict(torch.load(config["current_best_model"], map_location=torch.device('cpu')))

    pre_process = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])

    if not cap.isOpened():
        print("Error opening video stream or file")

    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        # process one frame over 3.
        if i % 3 == 0:
            if ret:
                # detect face on the frame
                face = crop_faces([frame])[0]
                if face is None:
                    out.write(frame)
                else:
                    (x, y, w, h) = face
                    # draw rectangle face delimiter
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))

                    # compute model predictions
                    pil_frame = Image.fromarray(frame)
                    pil_frame = pil_frame.resize(config["resolution"])  # TODO add that in pre-processing
                    x = pre_process(pil_frame).unsqueeze(0)
                    predictions = model.predict_single(x)

                    # write predictions on the output frame
                    for index, proba in enumerate(predictions):
                        text = "{}: {}%".format(config["catslist"][index], proba)
                        cv2.putText(frame, text, (10, 130 + 32 * index), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                    2, cv2.LINE_AA)

                    out.write(frame)
            else:
                break

        if i % 60 == 0:
            print("Processed {} seconds of video".format(i / 60))

    cap.release()
    out.release()
    cv2.destroyAllWindows()


make_video(20)
