import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..


features = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
    "Footage_to_Lot_Ratio",
    "Total_Rooms",
    "Age_of_House",
    "Garage_to_Footage_Ratio",
    "Avg_Room_Size",
    "House_Orientation_Angle",
    "Street_Alignment_Offset",
    "Solar_Exposure_Index",
    "Magnetic_Field_Strength",
    "Vibration_Level",
]

X_train = df_train[features]
y_train = df_train["Price"]
X_test = df_test[features]


# SUBTASK 1 :
# Pentru fiecare locuință din setul de testare,
# determinați suprafața totală estimată ca
# suma dintre suprafața casei (Square_Footage),
# dimensiunea garajului (Garage_Size)
# și dimensiunea terenului (Lot_Size).
df_subtask_1 = [
    (1, row["ID"], row["Square_Footage"] + row["Garage_Size"] + row["Lot_Size"])
    for _, row in df_test.iterrows()
]

# Subtask 2 (10 puncte)
# Pentru fiecare locuință din setul de testare,
# calculați raportul dintre dimensiunea garajului (Garage_Size)
# și numărul total de camere (Total_Rooms). Rezultatul va fi adăugat ca o nouă coloană numită Garage_to_Room_Ratio.
df_subtask_2 = [
    (2, row["ID"], row["Garage_Size"] / row["Total_Rooms"])
    for _, row in df_test.iterrows()
]
# aici am adaugat coloana
# asa oare?
df_test.assign(
    Garage_to_Room_Ratio=[
        row["Garage_Size"] / row["Total_Rooms"] for _, row in df_test.iterrows()
    ],
)

# SUBTASK 3
# Subtask 3 (10 puncte)
# Pentru fiecare locuință din setul de testare, calculați indicele de stabilitate a mediului,
# definit ca diferența dintre indexul de expunere solară (Solar_Exposure_Index)
# și nivelul vibrațiilor (Vibration_Level), împărțită la intensitatea câmpului magnetic (Magnetic_Field_Strength).
# Env_Stability_Index= (Solar_Exposure_Index - Vibration_Level) / Magnetic_Field_Strength​
df_subtask_3 = [
    (
        3,
        row["ID"],
        (row["Solar_Exposure_Index"] - row["Vibration_Level"])
        / row["Magnetic_Field_Strength"],
    )
    for _, row in df_test.iterrows()
]

# SUBTASK 4
# Pornind de la setul de date pentru antrenare,
# calculați media valorilor din coloana Square_Footage,
# reprezentând suprafața medie a caselor din setul de antrenare.
# Ulterior, pentru fiecare locuință din setul de testare,
# determinați modulul diferenței dintre suprafața efectivă a casei (Square_Footage) și media calculată pe setul de antrenare.
mean_square_footage = np.mean(df_train["Square_Footage"])
df_subtask_4 = [
    (4, row["ID"], abs(row["Square_Footage"] - mean_square_footage))
    for _, row in df_test.iterrows()
]

# Scopul principal al acestui task este de a construi un model de învățare automată
# care să prezică Price pe baza caracteristicilor furnizate în setul de date.
# Modelul trebuie să fie capabil să generalizeze bine pe date noi și este evaluat utilizând Mean Absolute Error (MAE).
# Implementați un model de regresie pentru predicția câmpului Price,
# utilizând setul de antrenament dataset_train.csv.
# Determinați predicțiile pe setul de evaluare ce este furnizat în fișierul CSV
# dataset_eval.csv (acesta nu conține coloana Price).
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

# corr = df_train[features + ["Price"]].corr()
# plt.figure(figsize=(14, 10))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Corelații cu Price")
# plt.show()
