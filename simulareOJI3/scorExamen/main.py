import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRFRegressor, XGBRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..


features = [
    # "ID",
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",  # trebuie ENCODING
    "Access_to_Resources",  # trebuie ENCODING
    "Extracurricular_Activities",  # trebuie ENCODING
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",  # trebuie ENCODING
    "Internet_Access",  # trebuie ENCODING
    "Tutoring_Sessions",
    "Family_Income",  # trebuie ENCODING
    "Teacher_Quality",  # trebuie ENCODING
    "School_Type",  # trebuie ENCODING
    "Peer_Influence",  # trebuie ENCODING
    "Physical_Activity",
    "Learning_Disabilities",  # trebuie ENCODING
    "Parental_Education_Level",  # trebuie ENCODING
    "Distance_from_Home",  # trebuie ENCODING
    "Gender",  # trebuie ENCODING
    # "Exam_Score",
]
features_to_be_encoded = [
    "Parental_Involvement",  # trebuie ENCODING
    "Access_to_Resources",  # trebuie ENCODING
    "Extracurricular_Activities",  # trebuie ENCODING
    "Motivation_Level",  # trebuie ENCODING
    "Internet_Access",  # trebuie ENCODING
    "Family_Income",  # trebuie ENCODING
    "Teacher_Quality",  # trebuie ENCODING
    "School_Type",  # trebuie ENCODING
    "Peer_Influence",  # trebuie ENCODING
    "Learning_Disabilities",  # trebuie ENCODING
    "Parental_Education_Level",  # trebuie ENCODING
    "Distance_from_Home",  # trebuie ENCODING
    "Gender",  # trebuie ENCODING
]

X_train = df_train[features]
y_train = df_train["Exam_Score"]
X_test = df_test[features]


# SUBTASK 1
# Pornind de la setul de date pentru antrenare, calculați media valorilor
# din coloana Hours_Studied, reprezentând numărul de ore alocate de studenți pentru studiu.
# Ulterior, pentru fiecare student din setul de testare,
# determinați modulul diferenței dintre numărul de ore efectiv studiate (Hours_Studied)
# și media calculată pe setul de antrenare.
mean_hours_studied = np.mean([row["Hours_Studied"] for _, row in df_train.iterrows()])
df_subtask_1 = [
    (1, row["ID"], abs(row["Hours_Studied"] - mean_hours_studied))
    for _, row in df_test.iterrows()
]

# SUBTASK 2
# Pe baza valorilor din coloana Sleep_Hours, determinați pentru fiecare student din setul de testare
# dacă acesta doarme un număr redus de ore.
# Considerăm că un student doarme puțin dacă are mai puțin de 7 ore de somn.
# Pentru studenții care dorm puțin vom scrie valoarea True,
# iar pentru cei care nu dorm puțin vom scrie valoarea False.
df_subtask_2 = [
    (2, row["ID"], "TRUE" if row["Sleep_Hours"] < 7 else "FALSE")
    for _, row in df_test.iterrows()
]

# SUBTASK 3
# Pentru fiecare student din setul de testare, determinați câți
# studenți din setul de antrenare au avut un scor anterior (Previous_Scores)
# mai mare sau egal decât al acelui student.
# putem face practic sume partiale pentru O(1) per query , adica in total sa avem O(n * m)
# n - lungimea la setul de training
# m - lungimea la setul de testing
# in total putem avea 0 de scoruri
pref_sum = [0] * 101  # [0.. 100]
vals_freq = [0] * 101
for _, row in df_train.iterrows():
    vals_freq[int(row["Previous_Scores"])] += 1
pref_sum[0] = vals_freq[0]
for i in range(1, 101):
    pref_sum[i] = pref_sum[i - 1] + vals_freq[i]

total = pref_sum[100]
df_subtask_3 = [
    (
        3,
        row["ID"],
        int(total - pref_sum[int(row["Previous_Scores"]) - 1])
        if int(row["Previous_Scores"]) > 0
        else total,
    )
    for _, row in df_test.iterrows()
]

# Pentru fiecare student din setul de testare,
# determinați numărul de studenți din setul de antrenare
# care au avut același nivel de motivație (Motivation_Level) ca al acelui student.
# doar facem frecventa la fiecare
motivation_freq = {"Low": 0, "Medium": 0, "High": 0}
for _, row in df_train.iterrows():
    motivation_freq[str(row["Motivation_Level"])] += 1

df_subtask_4 = [
    (
        4,
        row["ID"],
        int(motivation_freq.get(str(row["Motivation_Level"]), 0)),
    )
    for _, row in df_test.iterrows()
]


def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[
            (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
        ]
    return cleaned_df


# Facem Encoding
encoders = {}
for feature in features_to_be_encoded:
    le = LabelEncoder()

    # X_train[feature] = le.fit_transform(X_train[feature])
    X_train.loc[:, feature] = le.fit_transform(X_train[feature])
    # X_test[feature] = le.transform(X_test[feature])
    X_test.loc[:, feature] = X_test[feature].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    df_train[feature] = le.transform(df_train[feature])
    df_test[feature] = le.transform(df_test[feature])
    encoders[feature] = le
# am folosit ce zicea pe aici: https://stackoverflow.com/questions/47577168/how-can-i-increase-the-accuracy-of-my-linear-regression-modelmachine-learning
# df_train = remove_outliers_iqr(df_train, features)

X_train = df_train[features]
y_train = df_train["Exam_Score"]
X_test = df_test[features]

# 2 variante aici
# X_train = X_train.fillna(0)
# X_test = X_test.fillna(0)
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
y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
mae = mean_absolute_error(y_train, y_train_pred)
print(f"Training MSE: {mse:.4f}")
print(f"Training MAE: {mae:.4f}")

df_subtask_5 = pd.DataFrame(
    {
        "subtaskID": [5] * len(df_test),
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
df_subtask_3_df = pd.DataFrame(
    df_subtask_3, columns=["subtaskID", "datapointID", "answer"]
)
df_subtask_4_df = pd.DataFrame(
    df_subtask_4, columns=["subtaskID", "datapointID", "answer"]
)
df_subtask_5_df = pd.DataFrame(
    df_subtask_5, columns=["subtaskID", "datapointID", "answer"]
)

df_output = pd.concat(
    [
        df_subtask_1_df,
        df_subtask_2_df,
        df_subtask_3_df,
        df_subtask_4_df,
        df_subtask_5_df,
    ],
    ignore_index=True,
)

df_submission = pd.DataFrame(df_output, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("final_submission.csv", index=False)
print("Success")

# for feature in features_to_be_encoded:
#     le = LabelEncoder()
#     df_train[feature] = le.fit_transform(df_train[feature])
#     df_test[feature] = le.fit_transform(df_test[feature])
#
# corr = df_train[features + ["Exam_Score"]].corr()
# plt.figure(figsize=(14, 10))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# # TODO: add here the actual
# plt.title("Corelații cu Exam Score")
# plt.show()
