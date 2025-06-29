import pandas as pd
import json
import numpy as np
import ast


###### INITIAL , inainte sa salvam
# df_train = pd.read_csv("./train_data.csv", dtype={"id": str})
# df_train["pixels"] = df_train["pixels"].apply(ast.literal_eval)
# y_train = df_train["class"]
# X_train = np.array(df_train["pixels"].to_list(), dtype=np.float32)
# idxes = df_train["id"]
# np.save("./npy/X_train.npy", X_train)
# np.save("./npy/y_train.npy", y_train)
# np.save("./npy/indexes.npy", idxes)
######

# df_test = pd.read_csv("./test_data.csv", dtype={"id": str})
# df_test["pixels"] = df_test["pixels"].apply(ast.literal_eval)
# X_test = np.array(df_test["pixels"].to_list(), dtype=np.float32)
# np.save("./npy/X_test.npy", X_test)
X_train: np.ndarray = np.load("./npy/X_train.npy", allow_pickle=True)
y_train: np.ndarray = np.load("./npy/y_train.npy", allow_pickle=True)
X_test: np.ndarray = np.load("./npy/X_test.npy", allow_pickle=True)
idxes: np.ndarray = np.load("./npy/indexes.npy", allow_pickle=True)


print(type(X_train))  # <class 'numpy.ndarray'>
print(X_train.shape)  # (num_samples, vector_dim)
print(X_train[0][:1])  # primele 5 valori din primul vector

######
# mean_vector = np.mean(X_train, axis=0)  # face vectorul mediu pe coloana
# X_centered = X_train - mean_vector
# np.save("./npy/X_centered.npy", X_centered)
# np.save("./npy/mean_vector.npy", mean_vector)
######

X_centered: np.ndarray = np.load("./npy/X_centered.npy")
print(X_centered[:5])


### === Subtask 1

subtask1_rows = []
for idx, vec in zip(idxes, X_centered):
    json_vec = json.dumps(vec.tolist())
    subtask1_rows.append((1, idx, json_vec))
# for i, row in df_train.iterrows():
#     vector = np.array(row["pixels"])
#     answer_ls = [0] * (64 * 64)
#     answer_ls = np.array(answer_ls)
#     subtask1_rows.append((1, row["id"], answer_ls))

# # === Subtask 2
# X_test = np.vstack(df_test["pixels"].values)
# y_train = df_train["class"]
#
# y_pred = []  # values
subtask2_rows = []
# for id_, pred in zip(df_test["id"], y_pred):
#     subtask2_rows.append((2, id_, pred))
#
# === Final submission
submission_rows = subtask1_rows + subtask2_rows
df_submission = pd.DataFrame(
    submission_rows, columns=["subtaskID", "datapointID", "answer"]
)
df_submission.to_csv("submission.csv", index=False)
