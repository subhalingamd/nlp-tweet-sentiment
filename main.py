import sys
from utils import read_train_file, read_test_file, write_predictions
from preprocessing import preprocess
from constants import LENGTH_THRESH, RANDOM_SEED

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import FeatureUnion
# from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import PredefinedSplit

import pickle


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


def test(model_dir, in_path, out_path):
    X_test = read_test_file(in_path)

    try:
        model = None
        with open(model_dir+"/model.pkl", 'rb') as f:
            model = pickle.load(f)

        y_test = model.predict(X_test)
        y_test = [y*4 for y in y_test]

        write_predictions(out_path, y_test)

    except Exception as e:  # noqa: E722
        print("Could not load model. Generating random output.")
        print(e)
        write_predictions(out_path, [4]*len(X_test))


def train(data_dir, model_dir):

    data_path = "training.csv"
    if data_dir != "":
        data_path = data_dir + '/' + data_path

    X_train, y_train = read_train_file(data_path)
    X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_SEED)

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
            # ])),
            # ("classifier", LinearSVC(max_iter=10000, dual=False, random_state=RANDOM_SEED))
            ("classifier", LogisticRegression(
                solver="saga",
                penalty="l2",
                C=1.5,
                max_iter=10000,
                random_state=RANDOM_SEED,
                n_jobs=-1
                ))
            ], verbose=3)

    model = model.fit(X_train, y_train)

    with open(model_dir+"/model.pkl", 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    if len(sys.argv) < 2: raise ValueError("Missing mode (arg 1). Expecting [train|test]")
    mode = sys.argv[1]

    if mode == "train":
        data_dir = sys.argv[2]
        model_dir = sys.argv[3]
        train(data_dir=data_dir, model_dir=model_dir)
    elif mode == "test":
        model_dir = sys.argv[2]
        in_path = sys.argv[3]
        out_path = sys.argv[4]
        test(model_dir=model_dir, in_path=in_path, out_path=out_path)
    else:
        raise ValueError("Invalid mode (arg 1): '%s'. Expecting [train|test]" % mode)

