import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv("./train_data.csv")
df_test = pd.read_csv("./test_data.csv")

print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..

# aveam o problema ca unele numere aveau _ in ele, pe coloana asta

# TODO: doar copiezi outputul de coloane
features = [
    "Month",
    "Age",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    # "Type_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
]

features_to_be_encoded = [
    "Month",
    "Age",
    "Occupation",
    "Num_of_Loan",
    # "Type_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    # "Monthly_Balance",
]

X_train = df_train[features]
y_train = df_train["Credit_Score"]
X_test = df_test[features]


def clean_column(df, feature):
    df[feature] = df[feature].astype(str).str.replace("_", "", regex=True)
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    return df


X_train = clean_column(X_train, "Monthly_Balance")
X_test = clean_column(X_test, "Monthly_Balance")
X_train = clean_column(X_train, "Annual_Income")
X_test = clean_column(X_test, "Annual_Income")

X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# Only encode categorical columns, not numeric ones like 'Monthly_Balance'
encoders = {}
print("Before encoding")
for feature in features_to_be_encoded:
    if feature in ["Monthly_Balance", "Annual_Income"]:
        continue

    le = LabelEncoder()
    print(f"Processing {feature}")

    # Fit and transform on the train set
    X_train[feature] = le.fit_transform(X_train[feature])

    # Transform on the test set
    X_test[feature] = X_test[feature].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

    # Save the encoder
    encoders[feature] = le

print("After encoding")

# Now proceed with scaling

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Fit the model
model = LinearRegression()
print("Fitting Linear Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Predict
y_pred = model.predict(X_test)

# Prepare output
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
