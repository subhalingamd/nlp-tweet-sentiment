import re
from constants import REGEX

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
stopwords_en = set(stopwords.words("english"))

# The lemmatizer code is adapted from:
# https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
def nltk_tag_to_wordnet_tag(nltk_tag):  # convert nltk tag to wordnet tag
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize(sentence):
    # tokenize the sentence and find the POS tag for each token
    # nltk.word_tokenize(text)
    tokens = sentence.split()
    nltk_tagged = nltk.pos_tag(tokens)

    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:     # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:               # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return " ".join(lemmatized_sentence)


def to_lower(sentence):
    tokens = [token.lower() if not token.startswith('__') else token for token in sentence.split()]
    return " ".join(tokens)


def list_hashtag(s):
    return REGEX['hashtag'].findall(s)


def list_mention(s):
    return REGEX['mention'].findall(s)


def list_url(s):
    return REGEX['url'].findall(s)


def list_repeat(s):
    return REGEX['repeat'].findall(s)


def count_hashtag(s):
    return len(REGEX['hashtag'].findall(s))


def count_mention(s):
    return (REGEX['mention'].findall(s))


def count_url(s):
    return len(REGEX['url'].findall(s))


def count_repeat(s):
    return len(REGEX['url'].findall(s))


def process_hashtag(s):
    return REGEX['hashtag'].sub(lambda x: "__HASH__"+x.group(1).lower(), s)


def process_mention(s):
    return REGEX['mention'].sub(lambda x: "__AT__"+x.group(1).lower(), s)


def process_url(s):
    return REGEX['url'].sub("__URL__", s)


def process_repeat(s):
    return REGEX['repeat'].sub(r"\1\1", s)


def process_number(s):
    return REGEX['number'].sub(" ", s)


def preprocess(text):

    text = process_hashtag(text)
    text = process_mention(text)
    text = process_url(text)

    for symbol, name in REGEX['emotes'].items():
        text = text.replace(symbol, ' '+name+' ')

    for symbol, name in REGEX['punctuations'].items():
        text = text.replace(symbol, ' '+name+' ')

    text = text.replace("'", "")    # didn't
    text = REGEX['delimiter'].sub(' ', text)
    text = process_repeat(text)
    text = re.sub(r'\s+', ' ', text)
    text = to_lower(text)
    # text = text.replace("1", "one").replace("2", "to").replace("4", "for")
    text = process_number(text)


    text = " ".join([porter.stem(t) for t in text.split()])
    # text = lemmatize(text)
    return text


if __name__ == '__main__':
    # print(process_mention("@SubhalingamD @subhu2008 @subhalingamd"))
    # print(process_hashtag("#FuN #Subhalingam_D"))
    # print(process_url("https://www.github.com/SubhalingamD owns https://subhalingamd.me!"))
    # print(process_repeat("loool yessssss cool"))

    texts = [
                "@subhalingamd How are you.........??? :)", 
                "I'm #fine lollll!!", 
                "I don't know but this is better",
                "I'm trying harder to complete these assignments that have been given to me.... :)",
                "hungoverrr"
            ]
    for text in texts:
        print(text)
        print(preprocess(text))
