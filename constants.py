import re

RANDOM_SEED = 772

LENGTH_THRESH = 8

REGEX = {
    "hashtag": re.compile(r"#(\w+)"),
    "mention": re.compile(r"@(\w+)"),
    "url": re.compile(r"(?:http|https|ftp)://[a-zA-Z0-9\./]+"),
    "repeat": re.compile(r"(.)\1{2,}", flags=re.IGNORECASE),
    "delimiter": re.compile(r'\W+'),
    "number": re.compile(r"\s\d+\s"),
    "emotes": {},
    "punctuations": {}
}

EMOTES = [
            ('__EMOTE__SMILE',  [':-)', ':)', '(:', '(-:', ':3', ':-3', ]),
            ('__EMOTE__LAUGH',  [':-D', ':D', 'X-D', 'XD', 'xD',  '=D', '8D', '8-D', ':P', ':-P', ]),
            ('__EMOTE__LOVE',   ['<3', ':\\*',  '♥', ';^)', ':*', ':-*', ':X', '*_*',  ]),
            ('__EMOTE__WINK',   [';-)', ';)', ';-D', ';D', '(;', '(-;',  '*)', '*-)', ]),
            ('__EMOTE__SAD',    [':-(', ':(', '(:', '(-:',  ':<', ':-<', ':c', ':-\\', '(-;', ':/', ':-/', ]),        # ':/', ':-/' do not include??? #hack 
            ('__EMOTE__CRY',    [':,(', ':\'(', ':"(', ':((',  ':\'-(', ]),
            # TODO : https://en.wikipedia.org/wiki/List_of_emoticons
        ]
for name, symbols in EMOTES:
    for symbol in symbols:
        REGEX['emotes'].update({symbol: name})
        REGEX['emotes'].update({" ".join(symbol): name})
        if ')' in symbol:
            REGEX['emotes'].update({symbol.replace(')', ']'): name})
            REGEX['emotes'].update({" ".join(symbol).replace(')', ']'): name})
        if '(' in symbol:
            REGEX['emotes'].update({symbol.replace('(', '['): name})
            REGEX['emotes'].update({" ".join(symbol).replace('(', '['): name})

PUNCTUATIONS = [
                    ('__PUNC__EXCL',     ['!', '¡', ]),
                    ('__PUNC__QUES',     ['?', '¿', ]),
                    ('__PUNC__ELLP',     ['...', '…', '. . .', ]),
                    # TODO : http://en.wikipedia.org/wiki/Punctuation
                ]
for name, symbols in PUNCTUATIONS:
    for symbol in symbols:
        REGEX['punctuations'].update({symbol: name})


INTERNET_SLANG = {
    "u": "you",
    "ur": "your",
    "lol": "__EMOTE_LAUGH",
}
