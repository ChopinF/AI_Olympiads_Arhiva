import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..


features = [
    "LocalEarnings",
    "ChronoIndex",
    "RoomMetric",
    "SleepUnits",
    "TotalResidents",
    "DwellRatio",
    "GeoLat",
    "GeoLon",
]


df_train["Income_x_Rooms"] = df_train["LocalEarnings"] * df_train["RoomMetric"]
df_train["Rooms_per_Person"] = df_train["RoomMetric"] / df_train["DwellRatio"].replace(
    0, np.nan
)
df_train["Age_x_Rooms"] = df_train["ChronoIndex"] * df_train["RoomMetric"]
df_train["Earnings_per_Resident"] = df_train["LocalEarnings"] / df_train[
    "DwellRatio"
].replace(0, np.nan)
df_train["Lat_x_Lon"] = df_train["GeoLat"] * df_train["GeoLon"]
df_train["Lat_plus_Lon"] = df_train["GeoLat"] + df_train["GeoLon"]


df_test["Income_x_Rooms"] = df_test["LocalEarnings"] * df_test["RoomMetric"]
df_test["Rooms_per_Person"] = df_test["RoomMetric"] / df_test["DwellRatio"].replace(
    0, np.nan
)
df_test["Age_x_Rooms"] = df_test["ChronoIndex"] * df_test["RoomMetric"]
df_test["Earnings_per_Resident"] = df_test["LocalEarnings"] / df_test[
    "DwellRatio"
].replace(0, np.nan)
df_test["Lat_x_Lon"] = df_test["GeoLat"] * df_test["GeoLon"]
df_test["Lat_plus_Lon"] = df_test["GeoLat"] + df_test["GeoLon"]

new_features = [
    "LocalEarnings",
    "ChronoIndex",
    "RoomMetric",
    "SleepUnits",
    "TotalResidents",
    "DwellRatio",
    "GeoLat",
    "GeoLon",
    "Income_x_Rooms",
    "Rooms_per_Person",
    "Age_x_Rooms",
    "Earnings_per_Resident",
    "Lat_x_Lon",
    "Lat_plus_Lon",
]
# ID                  int64
# LocalEarnings     float64
# ChronoIndex       float64
# RoomMetric        float64
# SleepUnits        float64
# TotalResidents    float64
# DwellRatio        float64
# GeoLat            float64
# GeoLon            float64
# Price             float64
features_to_be_encoded = []

X_train = df_train[new_features]
y_train = df_train["Price"].clip(upper=np.percentile(df_train["Price"], 99))
X_test = df_test[new_features]


# am folosit ce zicea pe aici: https://stackoverflow.com/questions/47577168/how-can-i-increase-the-accuracy-of-my-linear-regression-modelmachine-learning
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_1 = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    loss="huber",
    random_state=42,
    verbose=1,
)

model_2 = LGBMRegressor(
    objective="regression_l1",
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
model_3 = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
stack = StackingRegressor(
    estimators=[("gbr", model_1), ("mlpr", model_2), ("xgb", model_3)],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1,
)


stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)

df_subtask_1 = pd.DataFrame(
    {
        "subtaskID": [1] * len(df_test),
        "datapointID": df_test["ID"].values,
        "answer": y_pred,
    }
)


df_subtask_1_df = pd.DataFrame(
    df_subtask_1, columns=["subtaskID", "datapointID", "answer"]
)
df_output = pd.concat(
    [
        df_subtask_1_df,
    ],
    ignore_index=True,
)

df_submission = pd.DataFrame(df_output, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("final_submission.csv", index=False)
print("Success")
