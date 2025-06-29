import pandas as pd
import cv2
from PIL import Image
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder


def mask_to_rle(mask):
    pixels = mask.T.flatten()
    rle = []
    in_run = False
    run_start = 0
    for i in range(len(pixels)):
        if pixels[i] == 1 and not in_run:
            in_run = True
            run_start = i + 1  # indexat de la 1
        elif pixels[i] == 0 and in_run:
            in_run = False
            rle.append(run_start)
            rle.append(i - run_start + 1)
    if in_run:
        rle.append(run_start)
        rle.append(len(pixels) - run_start + 1)
    return "[" + ", ".join(map(str, rle)) + "]"


results = []
directories = [
    "./arc/Satellite_Images-1/",
    "./arc/Satellite_Images-2/",
    "./arc/Satellite_Images-3/",
    "./arc/Satellite_Images-4/",
]


def process_dir(dir_name: str):
    for relative_path in os.listdir(dir_name):
        full_path = os.path.join(dir_name, relative_path)
        print(f"Processing {full_path}")
        image = cv2.imread(full_path)

        if image is None:
            print(f"Warning: could not read {full_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, color=1, thickness=cv2.FILLED)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        rle = mask_to_rle(mask)

        subtaskID = directories.index(dir_name) + 1
        results.append(
            {"subtaskID": subtaskID, "datapointID": relative_path, "answer": rle}
        )


for dir in directories:
    process_dir(dir)


df_output = pd.DataFrame(results)
df_output.to_csv("final_submission.csv", index=False)
