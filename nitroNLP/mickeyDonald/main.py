from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from string import punctuation

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from scipy.stats import mode

data_train = pd.read_csv("./train_data.csv")
data_test = pd.read_csv("./test_data.csv")


class Preprocess:
    def __init__(self, method="WordNetLemmatizer") -> None:
        self.method = method
        self.methods = [
            "LancesterStemmer",
            "PorterStemmer",
            "SnowballStemmer",
            "WordNetLemmatizer",
        ]
        if method not in self.methods:
            raise ValueError(
                f"The method should be from the following methods {self.methods}"
            )
        self.stuff_to_be_removed = list(stopwords.words("english")) + list(punctuation)
        self.stemmers = {
            "PorterStemmer": PorterStemmer(),
            "LancesterStemmer": LancasterStemmer(),
            "SnowballStemmer": SnowballStemmer(language="english"),
            "WordNetLemmatizer": WordNetLemmatizer(),
        }
        self.stemmer = self.stemmers[self.method]
        self.isFitted = False

    def preprocess(self, message: str) -> str:
        message = message.lower()
        message = re.sub(r"\$NE\$", "", message)
        message = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            message,
        )
        message = re.sub("(@[A-Za-z0-9_]+)", "", message)
        if self.method == "WordNetLemmatizer":
            message = " ".join(
                [
                    self.stemmer.lemmatize(word)
                    for word in message.split()
                    if word not in self.stuff_to_be_removed
                ]
            )
        else:
            message = " ".join(
                [
                    self.stemmer.stem(word)
                    for word in message.split()
                    if word not in self.stuff_to_be_removed
                ]
            )
        return message

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        X = X.apply(lambda x: self.preprocess(x))
        self.labelEncoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(X)
        self.labelEncoder.fit(y)
        self.isFitted = True

    def transform(self, X: pd.Series, y: pd.Series) -> tuple:
        if not self.isFitted:
            raise NotImplementedError("Please use fit function first")
        if isinstance(X, pd.Series):
            X = X.apply(lambda x: self.preprocess(x))
            vector = self.vectorizer.transform(X)
        else:
            X = self.preprocess(X)
            vector = self.vectorizer.transform([X])
        if y is not None:
            labels = self.labelEncoder.transform(y)
            return vector, labels
        else:
            return vector

    def fit_transform(self, X: pd.Series, y: pd.Series) -> tuple:
        self.fit(X, y)
        return self.transform(X, y)


def evaluate_model(modelName, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    trainPreds = model.predict(X_train)
    testPreds = model.predict(X_test)
    print("=" * 50, f"EVALUATING {modelName}", "=" * 50)
    print("Classification report on train set is")
    print(classification_report(y_true=y_train, y_pred=trainPreds))
    print("Classification report on test set is")
    print(classification_report(y_true=y_test, y_pred=testPreds))
    f1Train = f1_score(y_true=y_train, y_pred=trainPreds, average="weighted")
    f1Test = f1_score(y_true=y_test, y_pred=testPreds, average="weighted")
    return model, f1Train, f1Test


param_grid_log = {
    "penalty": ["l2"],
    "C": [0.1, 1, 10],
    "solver": ["liblinear"],
    "max_iter": [1000],
}

# param_grid_svc = {
#     "C": [0.01, 0.1, 1, 10, 100],
#     "loss": ["squared_hinge"],  # default, safest
#     "dual": [True, False],  # False often better when n_samples > n_features
#     "max_iter": [1000, 2000, 5000],
# }

param_grid_svc = {
    "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    "loss": ["hinge", "squared_hinge"],
    "dual": [True, False],
    "max_iter": [1000, 2000, 5000],
}

model_defs = [
    (LinearSVC(), param_grid_svc),
    (
        LogisticRegression(max_iter=1000),
        {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["liblinear"],
            "max_iter": [1000],
        },
    ),
    (
        RandomForestClassifier(),
        {
            "n_estimators": [100],
            "max_depth": [10, 20],
        },
    ),
]


def train_and_predict(label_column, method="WordNetLemmatizer"):
    print(f"\n--- Running for label: {label_column} ---\n")
    X = data_train["sample"]
    y = data_train[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = Preprocess(method=method)
    X_train_vec, y_train_enc = preprocessor.fit_transform(X_train, y_train)
    X_test_vec, y_test_enc = preprocessor.transform(X_test, y_test)

    models = []

    for base_model, grid in model_defs:
        cld = GridSearchCV(
            clone(base_model), param_grid=grid, cv=3, n_jobs=-1, verbose=True
        )
        cld.fit(X_train_vec, y_train_enc)
        best_model = cld.best_estimator_
        models.append(best_model)

        evaluate_model(
            base_model.__class__.__name__,
            best_model,
            X_train_vec,
            y_train_enc,
            X_test_vec,
            y_test_enc,
        )

    test_predictions = np.array([model.predict(X_test_vec) for model in models])
    final_preds = mode(test_predictions, axis=0).mode[0]

    answer = []
    for _, row in data_test.iterrows():
        vec = preprocessor.transform(row["sample"], None)
        preds = [model.predict(vec)[0] for model in models]
        voted = mode(preds, keepdims=True).mode[0]
        label = preprocessor.labelEncoder.inverse_transform([voted])[0]
        answer.append((row["datapointID"], label))

    if label_column == "dialect":
        results = [(1, idx, label) for idx, label in answer]
    else:
        results = [(2, idx, label) for idx, label in answer]

    df = pd.DataFrame(results, columns=["subtaskID", "datapointID", "answer"])
    return df


df_dialect = train_and_predict(label_column="dialect")
df_category = train_and_predict(label_column="category")

final_submission = pd.concat([df_dialect, df_category], ignore_index=True)

final_submission.to_csv("final_submission.csv", index=False)
print("Saved combined predictions to final_submission.csv")
