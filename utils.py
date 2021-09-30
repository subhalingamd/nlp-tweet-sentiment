def read_test_file(path):
    tweets = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            tweets.append(line)
        return tweets


def read_train_file(path):
    tweets, sentiments = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('"'):
                sentiment = 0 if line[1] == "0" else 1
                sentiments.append(sentiment)
                tweets.append(line[5:-1])
            else:
                tweets.append(line)
        return tweets, sentiments


def read_file(path):
    return read_train_file(path)


def data_split(X, y):
    from sklearn.model_selection import train_test_split
    from constants import RANDOM_SEED

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=RANDOM_SEED)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def read_file_and_split(path):
    X, y = read_file(path)
    return data_split(X, y)


def write_predictions(path, arr):
    with open(path, 'w', encoding='utf-8', errors='ignore') as f:
        print(*arr, sep='\n', file=f)
