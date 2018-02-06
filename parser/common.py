import numpy as np
import math
import time


def load_embeddings(filepath):
    with open(filepath, 'r') as embeddings_file:
        embeddings = embeddings_file.readlines()
    embeddings_sep = [emb.find(' ') for emb in embeddings]
    embeddings_words = [emb[:sep] for emb, sep in zip(embeddings, embeddings_sep)]
    embeddings_words = {w: i for i, w in enumerate(embeddings_words)}
    embeddings_values = [emb[sep + 1:-1].split(' ') for emb, sep in zip(embeddings, embeddings_sep)]
    embeddings_values = np.array(embeddings_values).astype(np.float)
    return embeddings_words, embeddings_values


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def fix_parentheses(query):
    imbalance = query.count('(') - query.count(')')
    if imbalance == 0:
        return query
    if imbalance > 0:
        return query + ' )' * imbalance
    if imbalance < 0:
        return query[:2 * imbalance]
    return query
