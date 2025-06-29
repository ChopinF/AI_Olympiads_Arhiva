import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline
from sklearn.ensemble import (
    StackingClassifier,
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from string import punctuation


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
df_test_subtask_1 = df_test[df_test["subtaskID"] == 1]
df_test_subtask_2 = df_test[df_test["subtaskID"] == 2]
print(len(df_test_subtask_2))

print(df_train.columns)
print(df_train.head)

# trebuie facut embedding pe text

X_train = df_train["text"]
y_train = df_train["label"]
X_test = df_test["text"]


class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=3,
            sublinear_tf=True,
        )
        self.labelEncoder = LabelEncoder()
        self.isFitted = False

    def basic_clean(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s" + re.escape(punctuation) + "]", "", text)
        return text

    def extract_custom_features(self, text):
        words = text.split()
        num_words = len(words) if len(words) > 0 else 1
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        punct_ratio = (
            len([c for c in text if c in punctuation]) / len(text)
            if len(text) > 0
            else 0
        )
        stopword_ratio = len([w for w in words if w in self.stop_words]) / num_words

        capital_ratio = (
            sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        )
        digit_ratio = (
            sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
        )

        word_counts = Counter(words)
        repeated_word_ratio = (
            sum(1 for w, count in word_counts.items() if count > 1) / num_words
        )

        return [
            avg_word_len,
            punct_ratio,
            stopword_ratio,
            capital_ratio,
            digit_ratio,
            repeated_word_ratio,
        ]

    def fit(self, X, y=None):
        X_cleaned = X.apply(self.basic_clean)
        self.vectorizer.fit(X_cleaned)
        if y is not None:
            self.labelEncoder.fit(y)
        self.isFitted = True
        return self

    def transform(self, X, y=None):
        if not self.isFitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        X_cleaned = X.apply(self.basic_clean)

        tfidf_features = self.vectorizer.transform(X_cleaned)

        custom_features = np.array([self.extract_custom_features(text) for text in X])

        combined = hstack([tfidf_features, custom_features])

        if y is not None:
            labels = self.labelEncoder.transform(y)
            return combined, labels
        else:
            return combined

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


### SUBTASK I
preproccessor = Preprocess()

X_train_vec, y_train_vec = preproccessor.fit_transform(X_train, y_train)

X_test_subtask_1 = df_test_subtask_1["text"]
X_test_vec_subtask_1 = preproccessor.transform(X_test_subtask_1, y=None)


model_1 = RandomForestClassifier(
    n_estimators=150,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=2,
    verbose=1,
    max_features="sqrt",
    max_depth=None,
    bootstrap=False,
)

model_2 = LinearSVC(max_iter=3000, loss="squared_hinge", C=10, random_state=42)

model_3 = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
stack = StackingClassifier(
    estimators=[("rfc", model_1), ("svc", model_2), ("xgb", model_3)],
    final_estimator=LogisticRegression(),
    n_jobs=-1,
)

stack.fit(X_train_vec, y_train_vec)
y_pred = stack.predict(X_test_vec_subtask_1)

df_subtask_1 = [(1, id, pred) for id, pred in zip(df_test["ID"], y_pred)]
df_subtask_1_df = pd.DataFrame(
    df_subtask_1, columns=["subtaskID", "datapointID", "answer"]
)

##### SUBTASK 2
# sa mor de stiu cum se face , bagam un llm

labels = ["SCIENCE", "BUSINESS", "CRIME", "RELIGION"]


model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ia 81 da ruleaza mai rapid
# model = SentenceTransformer("BAAI/bge-large-en-v1.5") # vreo 20-25 de minute, ia 86
label_embeddings = model.encode(labels)

preds = []


for idx, row in df_test_subtask_2.iterrows():
    text_embedding = model.encode(row["text"])
    similarities = model.similarity(text_embedding, label_embeddings)

    best_label = labels[np.argmax(similarities)]
    preds.append((2, row["ID"], best_label))
    print(f"Processed {row['ID'] - 10238 - 4763} /{len(df_test_subtask_2)}")

df_subtask_2_df = pd.DataFrame(preds, columns=["subtaskID", "datapointID", "answer"])

df_output = pd.concat(
    [
        df_subtask_1_df,
        df_subtask_2_df,
    ],
    ignore_index=True,
)


df_submission = pd.DataFrame(df_output, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("final_submission.csv", index=False)
print("Success")
