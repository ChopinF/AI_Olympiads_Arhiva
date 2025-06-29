import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier

df_train = pd.read_csv("./train_data.csv")
df_test = pd.read_csv("./test_data.csv")


features = [
    "X",
    "Y",
]

all_predictions = []
# facem un model separat pentru fiecare subtask in parte
for subtask_id in df_train["subtaskID"].unique():
    sub_train = df_train[df_train["subtaskID"] == subtask_id]
    sub_test = df_test[df_test["subtaskID"] == subtask_id]

    X_train = sub_train[features]
    y_train = sub_train["answer"]
    X_test = sub_test[features]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sub_result = pd.DataFrame(
        {
            "datapointID": sub_test["datapointID"],
            "answer": y_pred,
            "subtaskID": sub_test["subtaskID"],
        }
    )
    feature_importances = model.feature_importances_
    print(f"Feature Importances: {feature_importances}")

    all_predictions.append(sub_result)


df_submission = pd.concat(all_predictions)
df_submission.to_csv("final_submission.csv", index=False)
print("Success")
