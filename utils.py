import nltk, re
from string import punctuation

nltk.download('stopwords')
stops = nltk.corpus.stopwords.words('italian') + ['presso', 'dentro']
italian_punctuation = f"{punctuation}“”"

tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')

apostrophe = [
                "L’", "All’", "Nell’", "Dell’", "Dall’", "Coll’", "Sull’", "Un’",
                "l’", "all’", "nell’", "dell’", "dall’", "coll’", "sull’", "un’",
                "L'", "All'", "Nell'", "Dell'", "Dall'", "Coll'", "Sull'", "Un'",
                "l'", "all'", "nell'", "dell'", "dall'", "coll'", "sull'", "un'",
]


def clean_from_apostrophe(sentence):
    done = False
    for ap in apostrophe:
        if sentence.startswith(ap):
            if len(sentence[len(ap):]) > 0:
                sentence = sentence[len(ap):]
                done = True
    return sentence


def check_first_token(sentence):
    finish = False
    while not finish:
        words = sentence.split(' ', 1)
        if words[0].lower() in stops:
            if len(words) == 1:
                sentence = ""
                finish = True
            else:
                sentence = words[1]
        else:
            finish = True

    return sentence


def check_last_token(sentence):
    finish = False
    while not finish:
        words = sentence.rsplit(' ', 1)
        if words[-1].lower() in stops:
            if len(words) == 1:
                sentence = ""
                finish = True
            else:
                sentence = words[0]
        else:
            finish = True

    return sentence


def clean_italian_span(span):
    span = span.strip()

    span = check_first_token(span)
    span = check_last_token(span)
    return clean_from_apostrophe(span)


def check_symbols(token):
    symbols = ("#", "!","$", ")", "*", ",", "-", "/", ":", "%", "...",
                ";", "?", "@", "]", "^", "_", "}", "~", "''", "′", "£",
                "(", "{", "[", "``", "−", "”", "“", "«", "»",
                "Nell'", "Dell'", "Dall'", "Coll'", "Sull'", "Quell'", "Un'", "nell'",
                "dell'", "dall'", "coll'", "sull'", "quell'")

    check_l, check_all, check_full_stop = True, True, True
    found_symbols = []

    if token == "F.C.":
        return [token]

    for i, symbol in enumerate(symbols):
        if symbol in token:
            found_symbols.append((token.index(symbol), symbol))
            if symbol == "...":
                check_full_stop = False
            if symbol[-4:] == "all'":
                check_all = False
                check_l = False
            elif symbol[-2:] == "l'":
                check_l = False

    if "all'" in token and check_all:
        all_index = token.index("all'")
        if all_index == 0 or token[all_index] == " ":
            found_symbols.append((all_index, "all'"))
            check_l = False

    if "l'" in token and check_l:
        l_index = token.index("l'")
        if (l_index == 0 or token[l_index - 1] == " ") and (l_index == len(token) - 2 or token[l_index + 2] != "'"):
            found_symbols.append((l_index, "l'"))

    special_apostrophe = ("L'", "un'", "All'")
    for el in special_apostrophe:
        if el in token:
            l_index = token.index(el)
            if (l_index == 0 or token[l_index - 1] == " ") and (l_index == len(token) - len(el) or token[l_index + len(el)] != "'"):
                found_symbols.append((l_index, el))

    if "." in token and check_full_stop:
        found_symbols.append((token.index("."), "."))


    if "\"" in token:
        found_symbols += [(m.start(), "\"") for m in re.finditer("\"", token)]

    num_found_symbols = len(found_symbols)


    if num_found_symbols == 0:
        return [token]

    found_symbols = sorted(found_symbols)

    final_token = []
    for i, symbol in enumerate(found_symbols):
        before = token.split(symbol[1], 1)[0]
        if before != "":
            final_token.append(before)

        final_token.append(symbol[1])

        after = token.split(symbol[1], 1)[1]
        if after != "" and i == num_found_symbols-1:
            final_token.append(after)
        else:
            token = token[len(before) + len(symbol[1]):]

    return final_token


def tokenize_with_patterns(text):
    patterns = [
        r'^[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\s?[A-Z]\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\s?[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\s?[A-Z]\.\s?[A-Z]\.\s?(?:,|$)',
    ]

    elements = [(m.start(0), m.end(0)) for pattern in patterns for m in re.finditer(pattern, text)]

    if len(elements) == 0:
        return text.split()

    sorted_elements = sorted(elements, key=lambda x: x[0])

    tokenized_text = []
    start_index = 0

    for i, el in enumerate(sorted_elements):
        if start_index != el[0]:
            tokenized_text += text[start_index:el[0]].split()

        tokenized_text.append(text[el[0]: el[1]])

        if i == len(sorted_elements) - 1:
            tokenized_text += text[el[1]:].split()
        else:
            start_index = el[1]

    return tokenized_text


def check_patterns(text):
    patterns = [
        r'^[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\s?[A-Z]\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\s?[A-Z]\.\s?(?:,|$)',
        r'^[A-Z]\.\s?[A-Z]\.\s?[A-Z]\s?[A-Z]\.\s?[A-Z]\.\s?(?:,|$)',
    ]

    for pattern in patterns:
        if re.match(pattern, text):
            return True

    return False


def tokenize_italian_text(text):
    raw_sentences = tokenizer.tokenize(text.strip())

    tokenized_text = []
    for sentence in raw_sentences:
        tokenized_text += tokenize_with_patterns(sentence)

    refined_tokenization = []
    for el in tokenized_text:
        if check_patterns(el):
            refined_tokenization += el
        else:
            refined_tokenization += check_symbols(el)

    return refined_tokenization
