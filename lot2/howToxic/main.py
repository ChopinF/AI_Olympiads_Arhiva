import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


df_train: pd.DataFrame = pd.read_csv("./train_data.csv")
df_test: pd.DataFrame = pd.read_csv("./test_data.csv")
print(df_train.columns)  # pentru a avea toate coloanele
print(df_train.dtypes)  # pentru a afisa ce tip contine acea coloana int / float..
print(df_train.shape)
print(df_test.shape)


# Index(['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'insult'], dtype='object')
# id              object
# comment_text    object
# toxic            int64 -> label 0/1
# severe_toxic     int64 -> label 0/1
# obscene          int64 -> label 0/1
# insult           int64 -> label 0/1
feature_in = ["comment_text"]
features_out = ["toxic", "severe_toxic", "obscene", "insult"]
features_to_be_encoded = []


stop_words = stopwords.words("english")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class Preprocess:
    @staticmethod
    def remove_stopwords(text):
        no_stopword_text = [w for w in text.split() if w not in stop_words]
        return " ".join(no_stopword_text)

    @staticmethod
    def stemming(sentence):
        stemmer = SnowballStemmer("english")
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    @staticmethod
    def preprocess(series: pd.Series):
        series = series.apply(Preprocess.remove_stopwords)
        series = series.apply(Preprocess.stemming)
        return series


X_train = df_train[feature_in]
y_train = df_train[features_out]
X_test = df_test[feature_in]

X_train.loc[:, "comment_text"] = Preprocess.preprocess(X_train["comment_text"])
X_test.loc[:, "comment_text"] = Preprocess.preprocess(X_test["comment_text"])

X_train = X_train["comment_text"]
X_test = X_test["comment_text"]

print("Did preprocess")
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(),
        ),
        ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced"))),
    ]
)
pipeline.fit(X_train, y_train)
print("Did pipeline fit")
y_pred = pipeline.predict(X_test)
print("Did pipeline predict")

if issparse(y_pred):
    y_pred_dense = y_pred.toarray().tolist()
else:
    y_pred_dense = y_pred.tolist()


df_subtask = pd.DataFrame(
    {
        "subtaskID": [1] * len(df_test),
        "datapointID": df_test["id"].values,
        "answer": y_pred_dense,
    }
)


df_submission = pd.DataFrame(df_subtask, columns=["subtaskID", "datapointID", "answer"])

df_submission.to_csv("final_submission.csv", index=False)
print("Success")


# corr = df_train[features + [""]].corr()
# plt.figure(figsize=(14, 10))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# # TODO: adauga aici ce trebuie
# plt.title("Corelații cu ..")
# plt.show()

# aici am putut observa faptul ca data setul e foarte nebalansat, deci in model trebuie folosit "balanced"
# label_counts = df_train[features_out].sum().sort_values(ascending=False)
#
# plt.figure(figsize=(8, 5))
# sns.barplot(x=label_counts.index, y=label_counts.values)
# plt.title("Label Frequency Distribution")
# plt.ylabel("Number of Samples")
# plt.show()
