# -*- coding:utf-8 -*-
# Edited by bighead 19-1-31

import pandas as pd
import cv2
import numpy as np


def get_features_labels(file_path):
    """get the labels and features from the file in file_path, the features is the hog feature
    of the origin picture

    Args:
         file_path: the path of the data (train or test) ---- String

    return:
         labels: numpy array of the datas' labels ----numpy array
         features: numpy array of the pics' hog features ---- numpy array
    """
    data = pd.read_csv(file_path)
    data_csv = data.values

    labels = data_csv[:, 0]
    pixels = data_csv[:, 1:]

    features = []
    hog = cv2.HOGDescriptor("data/hog.xml")
    for img in pixels:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return (features, labels)

if __name__ == "__main__":
    get_features_labels("data/train.csv")
