from numpy import mean, std
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv("train_data.csv")
# print(df_train.head())
df_test = pd.read_csv("test_data.csv")

X_train = df_train[["distance_km", "package_weight_kg", "traffic_level"]]
y_train = df_train["on_time"]


X_test = df_test[["distance_km", "package_weight_kg", "traffic_level"]]
test_ids = df_test["id"]
# y_test = df_test["on_time"] # nu avem noi trebuie sa dam predict

# print(X_train.head())

model = LogisticRegression()
model.fit(X_train, y_train)

y_output = model.predict(X_test)


subtask1 = "mean_traffic_level"
subtask2 = "std_traffic_level"
subtask3 = "on_time"

# Defineste un model
# model
# Predictii model
# y_pred

# df_test[subtask3] = model.predict(df_test)

# Construiește rezultatul final
results = []

# Subtask 1 – un singur datapoint
# trebuie sa calculam mean_traffic_level
mean_traffic_level = round(mean(X_test["traffic_level"]), 2)
results.append((1, 1, mean_traffic_level))

# Subtask 2 – un singur datapoint
std_traffic_level = round(np.std(X_test["traffic_level"], ddof=1), 2)
results.append((2, 1, std_traffic_level))

# print(results)

# Subtask 3 – predicțiile pe test set
# for row in df_test.itertuples():
#     results.append((3, row.id, float(row.on_time)))

for idx, pred in zip(test_ids, y_output):
    results.append((3, idx, float(pred)))

# # Salveaza rezultatele
df_output = pd.DataFrame(results, columns=["subtaskID", "datapointID", "answer"])
df_output.to_csv("final_submission.csv", index=False)
