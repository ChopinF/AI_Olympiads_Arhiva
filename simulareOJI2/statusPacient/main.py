import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from numpy import array, mean, ndarray
from sklearn.ensemble import RandomForestClassifier

df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
features = [
    "Age",
    "T Stage",
    "N Stage",
    "6th Stage",
    "Grade",
    "Tumor Size",
    "Hemoglobin",
    "GFR",
    "Serum Creatinine",
    "BMI",
    "Heart Rate",
    "Serum Sodium",
    "Serum Potassium",
    "Serum Albumin",
    "Lactate",
    "Oxygen Saturation",
    "Blood Pressure",
    "Body Temperature",
]

X_train: pd.DataFrame = df_train[features]
print(df_train.columns)
X_test = df_test[features]
y_train: pd.DataFrame = df_train["Status"]
le_t_stage = LabelEncoder()
le_n_stage = LabelEncoder()
le_6th_stage = LabelEncoder()
le_grade = LabelEncoder()
# encode categorical features
X_train = X_train.assign(
    **{
        "6th Stage": le_6th_stage.fit_transform(X_train["6th Stage"]),
        "Grade": le_grade.fit_transform(X_train["Grade"]),
        "T Stage": le_t_stage.fit_transform(X_train["T Stage"]),
        "N Stage": le_n_stage.fit_transform(X_train["N Stage"]),
    }
)

X_test = X_test.assign(
    **{
        "6th Stage": le_6th_stage.transform(X_test["6th Stage"]),
        "Grade": le_grade.transform(X_test["Grade"]),
        "T Stage": le_t_stage.transform(X_test["T Stage"]),
        "N Stage": le_n_stage.transform(X_test["N Stage"]),
    }
)

# fill missing values with column means
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

### SUBTASK 1
df_subtask_1 = [
    (1, row["ID"], "Normal")
    if row["GFR"] >= 90
    else (1, row["ID"], "Mildly Decreased")
    if row["GFR"] >= 60
    else (1, row["ID"], "Decreased")
    for _, row in df_test.iterrows()
]

# ### SUBTASK 2 (Quartile pentru Serum Creatinine)
quart = df_train["Serum Creatinine"].quantile([0.25, 0.5, 0.75])
q1, q2, q3 = quart[0.25], quart[0.5], quart[0.75]

# clasificare pentru Serum Creatinine
df_subtask_2 = [
    (
        2,
        row["ID"],
        "Very Low"
        if row["Serum Creatinine"] <= q1
        else "Low"
        if row["Serum Creatinine"] <= q2
        else "High"
        if row["Serum Creatinine"] <= q3
        else "Very High",
    )
    for _, row in df_test.iterrows()
]

# ### SUBTASK 3 (Clasificare în funcție de media BMI)
mean_bmi = np.median(df_train["BMI"])
df_subtask_3 = [
    (3, row["ID"], 1 if row["BMI"] > mean_bmi else 0) for _, row in df_test.iterrows()
]

#  ### SUBTASK 4 (Numărul de pacienți cu același T Stage)
t_stage_counts = df_train["T Stage"].value_counts().to_dict()

# crearea rezultatelor pentru Subtask 4
df_subtask_4 = [
    (
        4,
        row["ID"],
        t_stage_counts.get(row["T Stage"], 0),
    )
    for _, row in df_test.iterrows()
]

## ### SUBTASK 5 (Model de clasificare)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Crearea rezultatelor pentru Subtask 5
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

df_submission.to_csv("submission.csv", index=False)
print("Success")
