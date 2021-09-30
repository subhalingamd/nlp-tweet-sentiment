import re

RANDOM_SEED = 772

LENGTH_THRESH = 8

REGEX = {
    "unicode": re.compile(r'((?:\\u[0-9A-Fa-f]+)|(?:[^\x00-\x84]))'),
    "hashtag": re.compile(r"#(\w+)"),
    "mention": re.compile(r"@(\w+)"),
    "url": re.compile(r"(?:https?|ftp)://[a-zA-Z0-9\./]+"),
    "repeat": re.compile(r"(.)\1{2,}", flags=re.IGNORECASE),
    "delimiter": re.compile(r'\W+'),
    "number": re.compile(r"(?:^|\W)\d+(?:\W|$)"),
    "emotes": {},
    "punctuations": {}
}

EMOTES = [
            ('__EMOTE__SMILE',  [':-)', ':)', '(:', '(-:', ':3', ':-3', ':P', ':-P', ':p', '^-^', '^_^',  ]),
            ('__EMOTE__LAUGH',  [':-D', ':D', 'X-D', 'XD', 'xD',  '=D', '8D', '8-D', 'X-p', '^.^', ':O', 'XO', ]), # :p ?
            ('__EMOTE__LOVE',   ['<3', ':\\*',  '♥', ';^)', ':*', ':-*', ':X', '*_*',  ]),
            ('__EMOTE__WINK',   [';-)', ';)', ';-D', ';D', '(;', '(-;',  '*)', '*-)', 'O.o', ]),
            ('__EMOTE__SAD',    [':-(', ':(', '):', ')-:',  ':<', ':-<', ':c', ':-\\', '(-;', ':/', ':-/', 'X-(', ':-@', 'O_O', ]),        # ':/', ':-/' do not include??? #hack 
            ('__EMOTE__CRY',    [':,(', ':\'(', ':"(', ':((',  ':\'-(', '>.<', ]),
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
                ]
for name, symbols in PUNCTUATIONS:
    for symbol in symbols:
        REGEX['punctuations'].update({symbol: name})


CONTRACTIONS = [
                            (r'won\'t', 'will not'),
                            (r'can\'t', 'can not'),
                            (r'i\'m', 'i am'),
                            (r'ain\'t', 'is not'),
                            (r'(\w+)\'ll', '\g<1> will'),
                            (r'(\w+)n\'t', '\g<1> not'),
                            (r'(\w+)\'ve', '\g<1> have'),
                            (r'(\w+)\'s', '\g<1> is'),
                            (r'(\w+)\'re', '\g<1> are'),
                            (r'(\w+)\'d', '\g<1> would'),
                            (r'&', 'and'),
                            (r'dont', 'do not'),
                            (r'wont', 'will not'),
                ]


escape_for_regex = {
                    '-': '\-',
                    '/': '\/',
                    ':': '\:',
                    ')': '\)', 
                    '@': '\@', 
                    '<': '\<',
                    ' ': '\s',
                    '=': '\=',
                    '_': '\_',
                    ';': '\;',
                    '[': '\[',
                    '^': '\^',
                    '*': '\*',
                    ']': '\]',
                    '(': '\(',
                    '\\': '\\\\',
                    '>': '\>',
                    ',': '\,',
                    '.': '\.',
                    '!': '\!',
                    '?': '\?',
                    '+': '\+'
                    }
