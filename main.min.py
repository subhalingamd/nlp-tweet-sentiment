import sys
from utils import read_file
from preprocessing import preprocess
from constants import LENGTH_THRESH, RANDOM_SEED

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.decomposition import TruncatedSVD

import pickle


def train(data_dir, model_dir):
    pass


if __name__ == '__main__':
    mode = sys.argv[1]
    data_path = sys.argv[2]
    (X_train, y_train) = read_file(data_path)
    X_train, y_train, X_test, y_test = X_train[:-5], y_train[:-5], X_train[-5:], y_train[-5:]

    # X_train, y_train = X_train[15000:20001], y_train[15000:20000]+[1]

    model = Pipeline(steps=[
            # ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(
                preprocessor=preprocess,
                ngram_range=(1, 2),
                max_df=0.75,
                min_df=5,
                )),
            #     # ("excl", ExclamationCount()),
            #     # ("qmark", QuestionMarkCount()),
            #     # ("url", UrlCount()),
            #     # ("mention", MentionCount()),
            # ])),
            # ("classifier", LinearSVC(max_iter=10000, dual=False, random_state=RANDOM_SEED))
            ("classifier", LogisticRegression(solver="lbfgs", C=1.5, max_iter=10000, random_state=RANDOM_SEED, n_jobs=-1))
            ], verbose=3)

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f1_score(y_test, y_pred, average=None))
    print(accuracy_score(y_test, y_pred))

    out_dir = "out/"
    with open(out_dir+"/model_lr_min.mlmodel", 'wb') as f:
        pickle.dump(model, f)
