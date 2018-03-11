import string

BLACK_LIST = string.punctuation.replace('%', '') + '\n'


def normalize(text,
              black_list=BLACK_LIST,
              vocab=None, lowercase=True, tokenize=False):
    if black_list:
        text = text.translate(string.maketrans(BLACK_LIST, ' ' * len(BLACK_LIST)))
    if lowercase:
        text = text.lower()
    if vocab:
        text = ' '.join([word for word in text.split() if word in vocab])
    if tokenize:
        return text.split()
    else:
        return ' '.join(text.split())
