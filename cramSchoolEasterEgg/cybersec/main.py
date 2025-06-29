import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df_train = pd.read_csv("./train_data.csv")
df_test = pd.read_csv("./test_data.csv")

df_test["hour"] = pd.to_datetime(df_test["Timestamp"]).dt.hour
task1 = pd.DataFrame(
    {
        "subtaskID": 1,
        "datapointID": df_test["ID"],
        "answer": df_test["hour"].apply(lambda h: "PM" if h >= 12 else "AM"),
    }
)
df_test.drop(columns=["hour"], inplace=True)


features = [
    "Suspicious_Port_Activity",
    "Traffic_Volume_Variation",
    "Packet_Length_Anomaly",
    "Malware_Score",
    "Threat_Level_Index",
    "User_Behavior_Score",
    "Geo_Dispersion",
    "Payload_Entropy",
    "Login_Attempts",
    "Device_Response_Time",
    "Session_Duration",
    "Packet_Retry_Rate",
    "Anomaly_Tendency",
]

df_train["hour"] = pd.to_datetime(df_train["Timestamp"]).dt.hour
df_test["hour"] = pd.to_datetime(df_test["Timestamp"]).dt.hour
features.append("hour")

X_train = df_train[features]
y_train = df_train["Attack Type"]
X_test = df_test[features]
idx = df_test["ID"]

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)

X_test = pd.DataFrame(imputer.transform(X_test), columns=features)
X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

# before, the first run
# rf = RandomForestClassifier(random_state=42)
# param_dist = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "bootstrap": [True, False],
# }
#
# search = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_dist,
#     n_iter=20,
#     scoring="accuracy",
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1,
# )
#
# search.fit(X_train, y_train)
# best_model = search.best_estimator_
# print(search.best_params_)
# y_pred = best_model.predict(X_test)

# after we found the right params, second run
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=1,
    max_depth=10,
    bootstrap=False,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


task2 = pd.DataFrame({"subtaskID": 2, "datapointID": idx, "answer": y_pred})

submission_df = pd.concat([task1, task2], ignore_index=True)
submission_df.to_csv("final_submission.csv", index=False)

print("Submission saved to final_submission.csv")
