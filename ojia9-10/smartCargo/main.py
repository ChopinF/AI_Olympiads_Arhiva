import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder


# TODO: switch if id / ID
df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..


# TODO: doar copiezi outputul de coloane
features = [
    "ID",
    "City A",  # oras de plecare
    "City B",  # oras in care ajung
    "Distance",
    "Time of Day",
    "Weather",  # doar la asta encoding
    "Traffic",
    "Road Quality",
    "Driver Experience",
    #     "deliver_time",
]

# ID                     int64
# City A                object
# City B                object
# Distance               int64
# Time of Day            int64
# Weather               object
# Traffic              float64
# Road Quality           int64
# Driver Experience      int64
# deliver_time           int64
# dtype: object

features_to_be_encoded = ["Weather"]

X_train = df_train[features]
y_train = df_train["deliver_time"]
X_test = df_test[features]

# Sarcina 1 (20p): Situația Bârlad: un client important din Bârlad a
# raportat întârzieri frecvente pe cursele în condiții de ceață.
# Managerul vrea să afle câte curse pleacă din Bârlad și pe vreme de ceață (Fog)
# în setul de date de predicție. Găsește și raportează numărul acestor curse.

print(df_train["City A"].unique())
barlad_fog_count = df_test[
    (df_test["City A"] == "Barlad") & (df_test["Weather"] == "Fog")
].shape[0]
df_subtask_1 = [(1, 1, barlad_fog_count)]


# df_subtask_2 = [(2, row["ID"], row[""]) for _, row in df_test.iterrows()]
#
# df_subtask_3 = [(3, row["ID"], row[""]) for _, row in df_test.iterrows()]
#
# df_subtask_4 = [(4, row["ID"], row[""]) for _, row in df_test.iterrows()]

features = [
    "ID",
    # "City A",  # oras de plecare
    # "City B",  # oras in care ajung
    "Distance",
    "Time of Day",
    "Weather",  # doar la asta encoding
    "Traffic",
    "Road Quality",
    "Driver Experience",
    #     "deliver_time",
]

X_train = df_train[features]
y_train = df_train["deliver_time"]
X_test = df_test[features]

encoders = {}
for feature in features_to_be_encoded:
    le = LabelEncoder()

    X_train.loc[:, feature] = le.fit_transform(X_train[feature])
    X_test.loc[:, feature] = X_test[feature].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    df_train[feature] = le.transform(df_train[feature])
    df_test[feature] = le.transform(df_test[feature])
    encoders[feature] = le

# am folosit ce zicea pe aici: https://stackoverflow.com/questions/47577168/how-can-i-increase-the-accuracy-of-my-linear-regression-modelmachine-learning
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

poly_degree = 2
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

df_subtask_2 = pd.DataFrame(
    {
        "subtaskID": [2] * len(df_test),
        "datapointID": df_test["ID"].values,
        "answer": y_pred,
    }
)


df_subtask_1_df = pd.DataFrame(
    df_subtask_1, columns=["subtaskID", "datapointID", "answer"]
)
df_subtask_2_df = pd.DataFrame(
    df_subtask_2, columns=["subtaskID", "datapointID", "answer"]
)
# df_subtask_3_df = pd.DataFrame(
#     df_subtask_3, columns=["subtaskID", "datapointID", "answer"]
# )
# df_subtask_4_df = pd.DataFrame(
#     df_subtask_4, columns=["subtaskID", "datapointID", "answer"]
# )
# df_subtask_5_df = pd.DataFrame(
#     df_subtask_5, columns=["subtaskID", "datapointID", "answer"]
# )

df_output = pd.concat(
    [
        df_subtask_1_df,
        df_subtask_2_df,
        # df_subtask_3_df,
        # df_subtask_4_df,
        # df_subtask_5_df,
    ],
    ignore_index=True,
)

df_submission = pd.DataFrame(df_output, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("final_submission.csv", index=False)
print("Success")

corr = df_train[features + ["deliver_time"]].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# TODO: adauga aici ce trebuie
plt.title("Corelații cu deliver_time")
plt.show()
