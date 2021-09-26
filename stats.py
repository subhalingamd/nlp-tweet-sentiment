import sys
from collections import Counter
from utils import read_file_and_split
from preprocessing import list_hashtag, list_mention, count_url, count_repeat
from wordcloud import WordCloud
from wordcloud import STOPWORDS as wc_stopwords
import matplotlib.pyplot as plt
import random
import re
import numpy as np


def class_distribution(data, labels):
    print(f" {Counter(labels)}")


def generate_wordcloud(data, freq=True, title=None, stopwords=None):
    wordcloud = WordCloud(stopwords=stopwords)
    if freq:
        wordcloud.generate_from_frequencies(frequencies=data)
    else:
        wordcloud.generate(data)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


def tweet_features(data, labels, gen_wordcloud=True):
    labels_set = set(labels)
    hashtags, mentions, urls, repeats = {}, {}, {}, {}
    num_hashtags, num_mentions, num_urls, num_repeats = 0, 0, 0, 0

    for s, l in zip(data, labels):
        hashtag = list_hashtag(s)
        mention = list_mention(s)
        url = count_url(s)
        repeat = count_repeat(s)
        for h in hashtag:
            if l not in hashtags:
                hashtags.update({l: {h.lower(): 1}})
            else:
                hashtags[l][h.lower()] = hashtags[l].get(h.lower(), 0) + 1
        for m in mention:
            if l not in mentions:
                mentions.update({l: {m.lower(): 1}})
            else:
                mentions[l][m.lower()] = mentions[l].get(m.lower(), 0) + 1
        urls[l] = urls.get(l, 0) + url
        repeats[l] = repeats.get(l, 0) + repeat

    print("-"*73)
    print("| {0:<6}  | {1:^17} | {2:^17} | {3:^8} | {4:^8} |".format(
        "labels", "hashtags", "mentions", "urls", "repeats"))
    print("| {0:<6}  | {1:^8} {2:^8} | {3:^8} {4:^8} | {5:^8} | {6:^8} |".format(
        "", "unique", "count", "unique", "count", "count", "count"))
    print("|-{0:<6}--+-{1:^17}-+-{2:^17}-+-{3:^8}-+-{4:^8}-|".format(
        "-"*6, "-"*17, "-"*17, "-"*8, "-"*8))

    for l in labels_set:
        print("| {0:^6}  | {1:^8} {2:^8} | {3:^8} {4:^8} | {5:^8} | {6:^8} |".format(  # noqa: E501
            l, len(hashtags.get(l, {})), sum(hashtags.get(l, {}).values()), len(mentions.get(l, {})), sum(mentions.get(l, {}).values()), urls.get(l, 0), repeats.get(l, 0)))  # noqa: E501
        num_hashtags += sum(hashtags.get(l, {}).values())
        num_mentions += sum(mentions.get(l, {}).values())
        num_urls += urls.get(l, 0)
        num_repeats += repeats.get(l, 0)

    print("|-{0:<6}--+-{1:^17}-+-{2:^17}-+-{3:^8}-+-{4:^8}-|".format(
        "-"*6, "-"*17, "-"*17, "-"*8, "-"*8))
    print("| {0:<6}  | {1:^17} | {2:^17} | {3:^8} | {4:^8} |".format(  # noqa: E501
            "all", num_hashtags, num_mentions, num_urls, num_repeats))  # noqa: E501
    print("-"*73)

    if gen_wordcloud:
        for l, h in hashtags.items():
            generate_wordcloud(h, title=f"Hashtags: {l}")
        for l, m in mentions.items():
            generate_wordcloud(m, title=f"Mentions: {l}")


def symbols_ditr(symbol, threshold=15, title=None):
    counter = {}
    for tweet, label in zip(X_train, y_train):
        count = tweet.count(symbol)
        if count > threshold:
            continue
        if label not in counter.keys():
            counter.update({label: {count: 1}})
        else:
            counter[label][count] = counter[label].get(count, 0) + 1
    max_count = 0
    for c in counter.values():
        max_count = max(max_count, max(c.keys()))
    x = np.arange(max_count+1)
    width = 1/len(counter)
    offset = 0
    for l, c in counter.items():
        for i in range(max_count+1):
            if i not in c.keys():
                c[i] = 0
        plt.bar(x + offset, c.values(), label=l, width=width)
        offset += width

    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data_path = sys.argv[1]
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = read_file_and_split(data_path)
    print()

    if False:
        print("******** Class Distribution ********")
        class_distribution(X_train, y_train)
        class_distribution(X_val, y_val)
        class_distribution(X_test, y_test)

    if False:
        print("\n\n******** Tweet features (Train) ********")
        tweet_features(X_train, y_train)

    if False:
        print("\n\n******** Tweet features (Val) ********")
        tweet_features(X_val, y_val)

    if False:
        generate_wordcloud(" ".join(X_train), title="X_train", stopwords=wc_stopwords, freq=False)

    if False:
        symb = set()
        for tweet in X_train:
            ss = re.findall(r"\W+", tweet)
            for s in ss:
                symb.add(s)
        print(symb)

    if True:
        symbols_ditr("!", title="Exclamations")
        symbols_ditr("?", title="Question Marks")
        symbols_ditr("http", title="URLs")
        symbols_ditr("#", title="Hashtags")
        symbols_ditr("@", title="Mentions")
        symbols_ditr("not", title="not")
        symbols_ditr("n't", title="n't")
