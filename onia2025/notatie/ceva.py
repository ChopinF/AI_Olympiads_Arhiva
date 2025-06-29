import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# maybe add some preprocessing
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
            img = img.convert("L")  # grayscale, shape will be (48, 48)
            return np.array(img)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {full_path}")
        return None


def segment_test_image(img, tile_size=48):
    h, w = img.shape
    n_tiles = w // tile_size
    tiles = [img[:, i * tile_size : (i + 1) * tile_size] for i in range(n_tiles)]
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
        "id": df_test_paths["datapointID"].apply(load_image),
    }
)

test_segments = []
test_ids = []
for idx, row in df_test.iterrows():
    tiles = segment_test_image(row["image"])
    for i, tile in enumerate(tiles):
        test_segments.append(tile.flatten())
        test_ids.append(f"{row['id']}_{i}")

print(df_train.head())

# TODO: doar copiezi outputul de coloane
features = ["image"]
features_to_be_encoded = []

X_train = df_train[features]
lengths = set()
for img in X_train["image"]:
    lengths.add(len(img))
print(lengths)

X_train = np.array([img.flatten() for img in X_train["image"]])
le = LabelEncoder()
y_train = df_train["label"]
y_train = le.fit_transform(y_train)
X_test_segmented = np.array(test_segments)

model = LogisticRegression(verbose=1)
model.fit(X_train, y_train)

y_pred_encoded = model.predict(X_test_segmented)
y_pred = le.inverse_transform(y_pred_encoded)

# df_subtask_1 = [(1, row["ID"], row[""]) for _, row in df_test.iterrows()]
#
# df_subtask_1_df = pd.DataFrame(
#     df_subtask_1, columns=["subtaskID", "datapointID", "answer"]
# )
#
# df_output = pd.concat(
#     [
#         df_subtask_1_df,
#         # df_subtask_2_df,
#         # df_subtask_3_df,
#         # df_subtask_4_df,
#         # df_subtask_5_df,
#     ],
#     ignore_index=True,
# )
#
# df_submission = pd.DataFrame(df_output, columns=["subtaskID", "datapointID", "answer"])
#
# df_submission.to_csv("final_submission.csv", index=False)
# print("Success")

# corr = df_train[features + [""]].corr()
# plt.figure(figsize=(14, 10))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# # TODO: adauga aici ce trebuie
# plt.title("Corelații cu ..")
# plt.show()
#
#
# #   1, -2, 1, 1, 1, -1,niciuna, 1, -1, -1.
# # 0 1, -1, 0, 1, 2,  1, 1     , 2,  1, 0
# #   1, -1, 0, 1, 2,  1, 1,      2,  1, 0
