import re
from constants import REGEX, CONTRACTIONS, escape_for_regex

# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

# from functools import lru_cache

# lemmatizer = WordNetLemmatizer()
# stopwords_en = set(stopwords.words("english"))
porter = PorterStemmer()

# store as variables -> optimization
REGEX__unicode = REGEX['unicode']
REGEX__hashtag = REGEX['hashtag']
REGEX__mention = REGEX['mention']
REGEX__url = REGEX['url']
REGEX__repeat = REGEX['repeat']
REGEX__delimiter = REGEX['delimiter']
REGEX__number = REGEX['number']

REGEX__emotes__dict = REGEX['emotes']
REGEX__punctuations__dict = REGEX['punctuations']
REGEX__emotes_punctuations__dict = {**REGEX__emotes__dict, **REGEX__punctuations__dict}

REGEX__emotes__keys = [x.translate(str.maketrans(escape_for_regex)) for x in REGEX['emotes'].keys()]
REGEX__punctuations__keys = [x.translate(str.maketrans(escape_for_regex)) for x in REGEX['punctuations'].keys()]
REGEX__emotes_punctuations__keys = REGEX__emotes__keys + REGEX__punctuations__keys

REGEX__emotes = re.compile(r'(' + '|'.join(REGEX__emotes__keys) + r')')
REGEX__punctuations = re.compile(r'(' + '|'.join(REGEX__punctuations__keys) + r')')
REGEX__emotes_punctuations = re.compile(r'(' + '|'.join(REGEX__emotes_punctuations__keys) + r')')

REGEX__contractions = [(re.compile(regex, flags=re.IGNORECASE), repl) for regex, repl in CONTRACTIONS]

REGEX__slangs__dict = {}
with open('slang.txt', 'r', encoding="utf-8") as f:
    REGEX__slangs__dict = dict(map(str.strip, line.partition('\t')[::2]) for line in f if line.strip())
SLANGS = sorted(REGEX__slangs__dict, key=len, reverse=True)
REGEX__slangs__keys = set(REGEX__slangs__dict.keys())
REGEX__slangs = re.compile(r"\b({})\b".format("|".join(map(re.escape, SLANGS))), flags=re.IGNORECASE)

REGEX__not_tag = re.compile(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', flags=re.IGNORECASE)
REGEX__not_tag_1 = re.compile(r'(\s+)(\w+)')


def REPLACE__hashtag(x) -> str: return "__HASH__"+x.group(1).lower()
def REPLACE__mention(x) -> str: return "__AT__"+x.group(1).lower()
def REPLACE__emotes(x) -> str: return " "+REGEX__emotes__dict.get(x.group())+" "
def REPLACE__punctuations(x) -> str: return " "+REGEX__punctuations__dict.get(x.group())+" "
def REPLACE__emotes_punctuations(x) -> str: return " "+REGEX__emotes_punctuations__dict.get(x.group())+" "
def REPLACE__slangs(x) -> str: return REGEX__slangs__dict.get(x.group(1).lower())
def REPLACE__not_tag(x) -> str: return REGEX__not_tag_1.sub(r'\1__NEG__\2', x.group(0))


def to_lower(sentence):
    # tokens = [token.lower() if not token.startswith('__') else token for token in sentence.split()]
    # return " ".join(tokens)
    return sentence.lower()


def list_unicode(s):
    return REGEX__unicode.findall(s)


def list_hashtag(s):
    return REGEX__hashtag.findall(s)


def list_mention(s):
    return REGEX__mention.findall(s)


def list_url(s):
    return REGEX__url.findall(s)


def list_repeat(s):
    return REGEX__repeat.findall(s)


def count_unicode(s):
    return len(REGEX__unicode.findall(s))


def count_hashtag(s):
    return len(REGEX__hashtag.findall(s))


def count_mention(s):
    return (REGEX__mention.findall(s))


def count_url(s):
    return len(REGEX__url.findall(s))


def count_repeat(s):
    return len(REGEX__repeat.findall(s))


def process_unicode(s):
    return REGEX__unicode.sub('', s)


def process_hashtag(s):
    return REGEX__hashtag.sub(REPLACE__hashtag, s)


def process_mention(s):
    return REGEX__mention.sub(REPLACE__mention, s)


def process_url(s):
    return REGEX__url.sub("__URL__", s)


def process_repeat(s):
    return REGEX__repeat.sub(r"\1\1", s)


def process_number(s):
    return REGEX['number'].sub(" ", s)


def process_emotes(s):
    # regex = r'(' + '|'.join(REGEX__emotes__keys) + r')'
    return REGEX__emotes.sub(REPLACE__emotes, s)


def process_punctuations(s):
    # regex = r'(' + '|'.join(REGEX__punctuations__keys) + r')'
    return REGEX__punctuations.sub(REPLACE__punctuations, s)


def process_emotes_and_punctuations(s):
    # regex = r'(' + '|'.join(REGEX__emotes_punctuations__keys) + r')'
    return REGEX__emotes_punctuations.sub(REPLACE__emotes_punctuations, s)


def process_contractions(text):
    for (pattern, repl) in REGEX__contractions:
        text = re.sub(pattern, repl, text)
    return text


def process_slangs(s):
    return REGEX__slangs.sub(REPLACE__slangs, s)


# #TODO:: modify -> use FWL?
def add_not_tag(s):     # #TODO broken... :/ #wontfix
    # ["not", "never", "no"] => __NEG__ prefix to words that follow, until *next punctuation*
    # Adapted from: https://github.com/Deffro/text-preprocessing-techniques
    return REGEX__not_tag.sub(REPLACE__not_tag, s)


def preprocess(text):
    try:
        text = text.strip()
        text = process_unicode(text)
        text = process_hashtag(text)
        text = process_mention(text)
        text = process_url(text)

        # for symbol, name in REGEX['emotes'].items():
        #     text = text.replace(symbol, ' '+name+' ')
        # text = process_emotes(text)

        # for symbol, name in REGEX['punctuations'].items():
        #     text = text.replace(symbol, ' '+name+' ')
        # text = process_punctuations(text)

        text = process_emotes_and_punctuations(text)

        text = process_repeat(text)
        text = process_contractions(text)
        text = process_slangs(text)

        text = text.replace("'", "")    # didn't #hack
        text = REGEX__delimiter.sub(' ', text)
        # text = re.sub(r'\s+', ' ', text)  # split() takes care #hack

        texts = text.split()
        text = []
        for word in texts:
            if len(word) > 3 and not word.startswith("__"):
                if word[-1] == word[-2] and word[:-1] in REGEX__slangs__keys:
                    text.append(process_slangs(word[:-1]))
                elif word[-3] == word[-2] and word[:-2]+word[-1] in REGEX__slangs__keys:
                    text.append(process_slangs(word[:-2]+word[-1]))
                elif word[-1] == word[-2] and word[-3] == word[-4] and word[:-3]+word[-1] in REGEX__slangs__keys:
                    text.append(process_slangs(word[:-3]+word[-2]))
                else:
                    text.append(word)
            else:
                text.append(word)
        text = " ".join(text)

        # text = text.replace("1", "one").replace("2", "to").replace("4", "for")
        text = process_number(text)

        text = add_not_tag(text)

        text = to_lower(text)

        text = " ".join([porter.stem(t) for t in text.split()])
        return text

    except:  # noqa: E722
        return text


if __name__ == '__main__':
    # print(process_mention("@SubhalingamD @subhu2008 @subhalingamd"))
    # print(process_hashtag("#FuN #Subhalingam_D"))
    # print(process_url("https://www.github.com/SubhalingamD owns https://subhalingamd.me!"))
    # print(process_repeat("loool yessssss cool"))

    texts = [
                "@subhalingamd How are you.........??? :) :P :// xD",
                "I'm #fine lolll !!",
                "I doN't know but this is better...",
                "I'm trying harder to complete these assignments that have been given to me.... :)",
                "hungoverrr \u122c \x84",
                "tommorow is holidayyy.. ♥ LoLLLL",
                "this is NOT a :(subhalingam i hope u r fine)"
            ]

    for text in texts:
        print(text)
        print(preprocess(text))

