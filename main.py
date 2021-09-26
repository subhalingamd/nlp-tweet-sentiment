import sys
from utils import read_file_and_split
from preprocessing import preprocess
import numpy as np
from constants import LENGTH_THRESH, RANDOM_SEED

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from qexpand import prob_rerank_method


class ExclamationCount(BaseEstimator, TransformerMixin):

    def transform(self, X, **transform_params):
        count = [[min(str(x).count("!"), LENGTH_THRESH)] for x in X]
        return count

    def fit(self, X, y=None, **fit_params):
        return self


class QuestionMarkCount(BaseEstimator, TransformerMixin):

    def transform(self, X, **transform_params):
        count = [[min(str(x).count("?"), LENGTH_THRESH)] for x in X]
        return count

    def fit(self, X, y=None, **fit_params):
        return self


class UrlCount(BaseEstimator, TransformerMixin):

    def transform(self, X, **transform_params):
        count = [[min(str(x).count("http"), LENGTH_THRESH//4)] for x in X]
        return count

    def fit(self, X, y=None, **fit_params):
        return self


class MentionCount(BaseEstimator, TransformerMixin):

    def transform(self, X, **transform_params):
        count = [[min(str(x).count("@"), LENGTH_THRESH//2)] for x in X]
        return count

    def fit(self, X, y=None, **fit_params):
        return self


if __name__ == '__main__':
    mode = sys.argv[1]
    data_path = sys.argv[2]
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = read_file_and_split(data_path)

    for i in range(50000):  # range(len(X_train)):
        X_train[i] = preprocess(X_train[i])
    for i in range(5000):  # in range(len(X_val)):
        X_val[i] = preprocess(X_val[i])
    for i in range(5000):  # in range(len(X_test)):
        X_test[i] = preprocess(X_test[i])


    grid_params = [
        {
            "features__tfidf__ngram_range": [(1, 2)],
            "features__tfidf__max_df": [0.75, 0.85, 1.0],
            "features__tfidf__min_df": [5, 7, 10],
            "classifier__penalty": ['l1', 'l2'],
            "classifier__C": [0.01, 0.1, 0.5, 1, 2, 10],
            "classifier__solver": ["saga"]
        },
        {
            "features__tfidf__ngram_range": [(1, 2)],
            "features__tfidf__max_df": [0.75, 0.85, 1.0],
            "features__tfidf__min_df": [5, 7, 10],
            "classifier__penalty": ['elasticnet'],
            "classifier__C": [0.01, 0.1, 0.5, 1, 2, 10],
            "classifier__solver": ["saga"],
            "classifier__l1_ratio": [0.2, 0.5, 0.8]
        },
        {
            "features__tfidf__ngram_range": [(1, 2)],
            "features__tfidf__max_df": [0.75, 0.85, 1.0],
            "features__tfidf__min_df": [5, 7, 10],
            "classifier__penalty": ['l2'],
            "classifier__C": [0.01, 0.1, 0.5, 1, 2, 10],
            "classifier__solver": ["lbfgs"]
        }
    ]

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1]*50000+[0]*5000

    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold=split_index)

    model = Pipeline(steps=[
            ("features", FeatureUnion([
                ("tfidf", TfidfVectorizer(
                    # preprocessor=preprocess
                    )),
                # ("excl", ExclamationCount()),
                # ("qmark", QuestionMarkCount()),
                # ("url", UrlCount()),
                # ("mention", MentionCount()),
            ])),
            # ("svd", TruncatedSVD(n_components=500, random_state=RANDOM_SEED)),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED))
            ], verbose=2)

    model = GridSearchCV(model, grid_params, cv=pds, return_train_score=True, verbose=3, n_jobs=-1)
    model = model.fit(X_train[:50000]+X_val[:5000], y_train[:50000]+y_val[:5000])

    print(model.best_estimator_)
    print("*"*10)
    print(model.cv_results_)

    y_pred = model.predict(X_test[:5000])
    print(f1_score(y_test[:5000],y_pred,average=None))
    print(accuracy_score(y_test[:5000],y_pred))

    y_pred = model.predict(X_train[:50000])
    print(f1_score(y_train[:50000],y_pred,average=None))
    print(accuracy_score(y_train[:50000],y_pred))
