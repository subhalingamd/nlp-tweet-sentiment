import re
from constants import REGEX, CONTRACTIONS, escape_for_regex

# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from functools import partial
from collections import Counter
from functools import lru_cache

# lemmatizer = WordNetLemmatizer()
# stopwords_en = set(stopwords.words("english"))
porter = PorterStemmer()


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
with open('slang.txt') as f:
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


# @TODO:: modify

""" Replaces contractions from a string to their equivalents """
def add_not_tag(s):
    """ Finds "not,never,no" and adds the tag NEG_ to all words that follow until the next punctuation """
    return REGEX__not_tag.sub(REPLACE__not_tag, s)

'''
### Spell Correction begin ###
""" Spell Correction http://norvig.com/spell-correct.html """
def words(text): return re.findall(r'\w+', text.lower())
# WORDS = Counter(words(open('corporaForSpellCorrection.txt').read()))
# spell_voab_size = sum(WORDS.values())
# WORDS_keys = set(WORDS.keys())
WORDS_keys = set(slang_map.keys())
spell_voab_size = len(WORDS_keys)
def P(word, N=spell_voab_size): 
    """P robability of `word`. """
    return 1
@lru_cache(maxsize=512)
def spellCorrection(word): 
    """ Most probable spelling correction for word. """
    return max(candidates(word.lower()), key=P)
def candidates(word): 
    """ Generate possible spelling corrections for word. """
    if word in WORDS_keys:
        return [word]
    return (known(edits1(word)) or known(edits2(word)) or [word])
def known(words): 
    """ The subset of `words` that appear in the dictionary of WORDS. """
    return set(w for w in words if w in WORDS_keys)
letters    = 'abcdefghijklmnopqrstuvwxyz'
def edits1(word):
    """ All edits that are one edit away from `word`. """
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    """ All edits that are two edits away from `word`. """
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """
    repl = r'\1\2\3'
    if wordnet.synsets(word) or word.lower() in WORDS_keys:
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:      
        return replaceElongated(repl_word)
    else:       
        return repl_word
'''


def preprocess(text):
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

    # texts = [spellCorrection(w) if not w.startswith("__") else w for w in text.split()]
    text = " ".join([porter.stem(t) for t in text.split()])
    # text = lemmatize(text)
    return text


if __name__ == '__main__':
    # print(process_mention("@SubhalingamD @subhu2008 @subhalingamd"))
    # print(process_hashtag("#FuN #Subhalingam_D"))
    # print(process_url("https://www.github.com/SubhalingamD owns https://subhalingamd.me!"))
    # print(process_repeat("loool yessssss cool"))

    print(REGEX__emotes_punctuations__keys)

    texts = [
                "@subhalingamd How are you.........??? :) :P :// xD", 
                "I'm #fine lolll !!", 
                "I don't know but this is better",
                "I'm trying harder to complete these assignments that have been given to me.... :)",
                "hungoverrr \u122c \x84",
                "tommorow is holidayyy.. ♥ LoLLLL",
                "this is :(subhalingam i hope you are fine)"
            ]*1

    for text in texts:
        print(text)
        print(preprocess(text))

    print(process_contractions("i'm"))
