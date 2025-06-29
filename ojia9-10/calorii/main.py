import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..


features = [
    # "User_ID",
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Gender",
    # "Calories",
]
features_to_be_encoded = ["Gender"]

X_train = df_train[features]
y_train = df_train["Calories"]
X_test = df_test[features]

# Subtask 1: Samples: numărul de linii din setul de date de antrenare
df_subtask_1 = [(1, 1, df_train.shape[0])]

# Subtask 2: No.Males: numărul de exemple de antrenament care descriu activități realizate de către bărbați
df_subtask_2 = [(2, 1, (df_train["Gender"] == "male").sum())]

# Subtask 3: AverageDuration: durata medie a activităților din setul de antrenare.
# df_subtask_3 = [(3, 1, np.mean([row["Duration"] for _, row in df_train.iterrows()]))]
df_subtask_3 = [(3, 1, df_train["Duration"].mean())]

# Subtask 4: SeniorUsers: numărul de utilizatori, din setul de antrenare, care au cel puțin 75 de ani împliniți
df_subtask_4 = [(4, 1, (df_train["Age"] >= 75).sum())]


# Subtask 5: (60p) Dezvoltați un model de AI/ML și efectuați predicția pentru atributul Calories pentru fiecare exemplu din fișierul cu datele de test, pentru care coloana Subtask are valoarea 5.
encoders = {}
for feature in features_to_be_encoded:
    le = LabelEncoder()

    X_train.loc[:, feature] = le.fit_transform(X_train[feature])
    X_test.loc[:, feature] = X_test[feature].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    # df_train[feature] = le.transform(df_train[feature])
    # df_test[feature] = le.transform(df_test[feature])
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

df_subtask_5 = pd.DataFrame(
    {
        "subtaskID": [5] * len(df_test),
        "datapointID": df_test["User_ID"].values,
        "answer": y_pred,
    }
)

# Subtask 6: (20p) O echipă de handbal masculin are nevoie să estimeze consumul caloric pentru a optimiza dieta jucătorilor. Dezvoltați un model de AI/ML și realizați predicția numărului de calorii pentru pentru fiecare exemplu din fișierul cu datele de test pentru care coloana Subtask are valoarea 6


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
