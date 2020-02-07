import pandas as pd
from PIL import Image
import urllib
import sys
import numpy as np
import cv2
from uuid import uuid4
import traceback

from pipeline import crop_faces, crop_cv_img

"""
TODO
- write a script to:
    - parse all humans annotations and take the majority label
    - in a "triplets" dataframe, save
        triplet_id / im1_id / im2_id / im3_id / triplet_type / majority_label / majority_confidence
"""

dataframe = pd.read_csv("./fec/train.csv", names=[str(i) for i in range(50)])

SAVED_FACES_PATH = "./fec/train"
SAVED_DF_PATH = "./fec"


def add_img_to_faces_df(df, id, url, is_valid):
    return df.append({
            "uuid": id,
            "url": url,
            "is_valid": is_valid
        },
        ignore_index=True
    )


def downloader(fec_df):
    faces_df = pd.DataFrame(columns=["uuid", "url", "is_valid"])
    for index, row in fec_df.iterrows():
        if index % 50 == 0:
            print(index)
        for col in range(0, 15, 5):
            url = row[col]
            x1, x2, y1, y2 = float(row[col + 1]), float(row[col + 2]), float(row[col + 3]), float(row[col + 4])
            if url not in faces_df["url"].values:
                try:
                    img = Image.open(urllib.request.urlopen(url))
                    cv_img = np.array(img)
                    shape = cv_img.shape
                    x1, x2 = int(shape[1] * x1), int(shape[1] * x2)
                    y1, y2 = int(shape[0] * y1), int(shape[0] * y2)
                    annotated_face = cv_img[y1:y2, x1:x2]

                    [face_coords] = crop_faces([annotated_face])
                    if face_coords is not None:
                        (x, y, w, h) = face_coords
                        face = crop_cv_img(annotated_face, x, y, w, h)
                        face_id = uuid4()
                        cv2.imwrite("{}/{}.jpg".format(SAVED_FACES_PATH, face_id), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                        faces_df = add_img_to_faces_df(faces_df, face_id, url, True)
                    else:
                        faces_df = add_img_to_faces_df(faces_df, uuid4(), url, False)
                except urllib.error.HTTPError:
                    faces_df = add_img_to_faces_df(faces_df, uuid4(), url, False)
                except Exception:
                    _, error, tb = sys.exc_info()
                    print("UNEXPECTED ERROR:", error)
                    traceback.print_tb(tb)

    faces_df.to_csv("{}/faces.csv".format(SAVED_DF_PATH), index=False)


downloader(dataframe)
