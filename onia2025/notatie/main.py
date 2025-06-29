import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def preprocess_image(img_data):
    pass


df_train_paths: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test_paths: pd.DataFrame = pd.read_csv("./test_data.csv")


print(df_train_paths.columns)  # pentru a avea toate coloanele
print(df_train_paths.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..

print(df_test_paths.columns)
print(df_test_paths.dtypes)

print(df_train_paths["Path"].head())


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_image(relative_path):
    full_path = os.path.join(BASE_DIR, "starting_kit", relative_path)
    try:
        with Image.open(full_path) as img:
            img = img.convert("L")  # e bine in gray scale , top
            return np.array(img)
    except FileNotFoundError:
        print(f"[ERROR] File not found : {full_path}")
        return None


def segment_test_image(img, tile_size=48):
    h, w = img.shape
    number_tiles = w // tile_size  # impartim lugnimea la lungimea la un singur simbol
    tiles = [img[:, i * tile_size : (i + 1) * tile_size] for i in range(number_tiles)]
    print(f"Segmenting image {row['id']} into {len(tiles)} tiles")
    return tiles


df_train = pd.DataFrame(
    {
        "image": df_train_paths["Path"].apply(load_image),
        "label": df_train_paths["Effect"],
    }
)

print(df_train.head())


df_test = pd.DataFrame(
    {
        "image": df_test_paths["datapointID"].apply(load_image),
        "id": df_test_paths["datapointID"],
    }
)

test_segments = []
test_ids = []
for idx, row in df_test.iterrows():
    tiles = segment_test_image(row["image"])
    for i, tile in enumerate(tiles):
        test_segments.append(tile.flatten())
        test_ids.append(f"{row['id']}_{i}")


# X_train = df_train["image"]

X_train = np.array([img.flatten() for img in df_train["image"]])
le = LabelEncoder()
y_train = df_train["label"]
y_train = le.fit_transform(y_train)
X_test_segmented = np.array(test_segments)

model = LogisticRegression(verbose=1)
model.fit(X_train, y_train)

y_pred_encoded = model.predict(X_test_segmented)
y_pred = le.inverse_transform(y_pred_encoded)

print(y_pred)
