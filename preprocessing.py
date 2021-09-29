import re
from constants import REGEX

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from functools import partial
from collections import Counter
from functools import lru_cache

# lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
# stopwords_en = set(stopwords.words("english"))

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


def list_unicode(s):
    return REGEX['unicode'].findall(s)


def list_hashtag(s):
    return REGEX['hashtag'].findall(s)


def list_mention(s):
    return REGEX['mention'].findall(s)


def list_url(s):
    return REGEX['url'].findall(s)


def list_repeat(s):
    return REGEX['repeat'].findall(s)


def count_unicode(s):
    return len(REGEX['unicode'].findall(s))


def count_hashtag(s):
    return len(REGEX['hashtag'].findall(s))


def count_mention(s):
    return (REGEX['mention'].findall(s))


def count_url(s):
    return len(REGEX['url'].findall(s))


def count_repeat(s):
    return len(REGEX['url'].findall(s))


def process_unicode(s):
    return REGEX['unicode'].sub('', s)


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


# @TODO:: modify

""" Replaces contractions from a string to their equivalents """
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),                       (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text
def addNotTag(text):
    """ Finds "not,never,no" and adds the tag NEG_ to all words that follow until the next punctuation """
    transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', 
       lambda match: re.sub(r'(\s+)(\w+)', r'\1__NEG__\2', match.group(0)), 
       text,
       flags=re.IGNORECASE)
    return transformed
""" Creates a dictionary with slangs and their equivalents and replaces them """
with open('slang.txt') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True) # longest first for regex
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))), re.IGNORECASE)
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1).lower()])
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

# def P(word, N=spell_voab_size): 
#     """P robability of `word`. """
#     return WORDS[word] / N
# @lru_cache(maxsize=512)
# def spellCorrection(word): 
#     """ Most probable spelling correction for word. """
#     return max(candidates(word), key=P)
# def candidates(word): 
#     """ Generate possible spelling corrections for word. """
#     if word in WORDS_keys:
#         return [word]
#     return (known(edits1(word)) or known(edits2(word)) or [word])
# def known(words): 
#     """ The subset of `words` that appear in the dictionary of WORDS. """
#     return set(w for w in words if w in WORDS_keys)

# def edits1(word):
#     """ All edits that are one edit away from `word`. """
#     letters    = 'abcdefghijklmnopqrstuvwxyz'
#     splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#     deletes    = [L + R[1:]               for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#     replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#     inserts    = [L + c + R               for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)

# def edits2(word): 
#     """ All edits that are two edits away from `word`. """
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))

### Spell Correction End ###


def process_repeat(s, x):
    return REGEX['repeat'].sub(r"\1\1", s)
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

def preprocess(text):

    text = process_unicode(text)
    text = process_hashtag(text)
    text = process_mention(text)
    text = process_url(text)

    for symbol, name in REGEX['emotes'].items():
        text = text.replace(symbol, ' '+name+' ')

    for symbol, name in REGEX['punctuations'].items():
        text = text.replace(symbol, ' '+name+' ')

    text = process_repeat(text,2)
    text = replaceContraction(text)
    # text = " ".join([replaceElongated(w) if not w.startswith("__") else w for w in text.split()])
    text = replaceSlang(text)
    text = addNotTag(text)

    text = text.replace("'", "")    # didn't
    text = REGEX['delimiter'].sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = to_lower(text)
    # text = text.replace("1", "one").replace("2", "to").replace("4", "for")
    text = process_number(text)

    # texts = text.split()
    # text = []
    # for word in texts:
    #     if len(word) > 3 and not word.startswith("__"):
    #         if word[-1]==word[-2] and word[:-1] in WORDS_keys:
    #             text.append(replaceSlang(word[:-1]))
    #         elif word[-3]==word[-2] and word[:-2]+word[-1] in WORDS_keys:
    #             text.append(replaceSlang(word[:-2]+word[-1]))
    #         elif word[-1]==word[-2] and word[-3]==word[-4] and word[:-3]+word[-1] in WORDS_keys:
    #             text.append(replaceSlang(word[:-3]+word[-2]))
    #         else:
    #             text.append(word)
    #     else:
    #         text.append(word)

    # texts = [spellCorrection(w) if not w.startswith("__") else w for w in text.split()]
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
                "I'm #fine lolll !!", 
                "I don't know but this is better",
                "I'm trying harder to complete these assignments that have been given to me.... :)",
                "hungoverrr \u122c \x84",
                "tommorow is holyday.. LoL",

            ]
    for text in texts:
        print(text)
        print(preprocess(text))

    print(replaceContraction("tshouldn't"))
