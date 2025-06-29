import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    StackingClassifier,
)
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)
print(df_train.dtypes)

features = [
    "NT",  # nota la testul NT
    "MEV",  # nota la testul MEV
    "MATE",  # nota la testul MATE
    "MGIM",  # nota la testul MGIM
]

X_train = df_train[features]
y_train = df_train["status_admitere"]
X_test = df_test[features]

# subtask1 (20 puncte): dif_NT-MEV
# - reprezentând diferența dintre nota la testul de admitere NT și nota la evaluarea națională MEV a candidatului,
# precizie de 2 decimale (valori diferite pentru fiecare rând).
df_subtask_1 = [
    (1, row["id"], round(row["NT"] - row["MEV"], 2)) for _, row in df_test.iterrows()
]

# subtask2 (20 puncte): loc-MEV –
# reprezentând poziția în clasament a candidaților în funcție de
# nota obținută la evaluarea națională MEV (valori întregi diferite pe același rând).
scores = list(zip(df_test["MEV"], df_test["id"]))
sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
ranks = {v[1]: idx + 1 for idx, v in enumerate(sorted_scores)}

df_subtask_2 = [(2, row["id"], int(ranks[row["id"]])) for _, row in df_test.iterrows()]

# SUBTASK 3
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

poly_degree = 2
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

grid_linear_regression = {
    "copy_X": [True, False],
    "fit_intercept": [True, False],
    "n_jobs": [1, 5, 10, 15, None],
    "positive": [True, False],
}

# model = RandomForestClassifier()
# grid_random_forest_classifier = {
#     "bootstrap": [True, False],
#     "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
#     "max_features": ["sqrt", "log2", None],
#     "min_samples_leaf": [1, 2, 4],
#     "min_samples_split": [2, 5, 10],
#     "n_estimators": [130, 180, 230],
# }

# first run
# model = XGBClassifier()
#
# grid_boost = {
#     "max_depth": stats.randint(3, 10),
#     "learning_rate": stats.uniform(0.01, 0.1),
#     "subsample": stats.uniform(0.5, 0.5),
#     "n_estimators": stats.randint(50, 200),
# }
#
# clf = RandomizedSearchCV(model, param_distributions=grid_boost)
# # # fit
# clf.fit(X_train, y_train)
# model = clf.best_estimator_
# print(clf.best_params_)

model_1 = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=2,
    verbose=1,
    max_features="sqrt",
    max_depth=None,
    bootstrap=False,
)

model_2 = DecisionTreeClassifier()
model_3 = XGBClassifier(
    learning_rate=np.float64(0.06236430060502772),
    max_depth=6,
    n_estimators=109,
    subsample=np.float64(0.9834079972282441),
)

stack = StackingClassifier(
    estimators=[("rfc", model_1), ("dtc", model_2), ("xgb", model_3)],
    final_estimator=LogisticRegression(),
    n_jobs=-1,
)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

df_subtask_3 = pd.DataFrame(
    {
        "subtaskID": [3] * len(df_test),
        "datapointID": df_test["id"].values,
        "answer": y_pred,
    }
)

df_subtask_1_df = pd.DataFrame(
    df_subtask_1, columns=["subtaskID", "datapointID", "answer"]
)
df_subtask_2_df = pd.DataFrame(
    df_subtask_2, columns=["subtaskID", "datapointID", "answer"]
)
df_subtask_3_df = pd.DataFrame(
    df_subtask_3, columns=["subtaskID", "datapointID", "answer"]
)

df_output = pd.concat(
    [
        df_subtask_1_df,
        df_subtask_2_df,
        df_subtask_3_df,
    ],
    ignore_index=True,
)

df_submission = pd.DataFrame(df_output, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("final_submission.csv", index=False)
print("Success")

# corr = df_train[features + ["status_admitere"]].corr()
# plt.figure(figsize=(14, 10))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Corelai cu Price")
# plt.show()
