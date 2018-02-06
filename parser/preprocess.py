import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import re
from language import *


def normalize_target(target):
    target = re.sub(r"([,\(\)])", r" \1 ", target).strip()
    target = re.sub(r"  ", r" ", target)
    return target


def load_data(filepath):
    with open(filepath) as f:
        s = f.read()
    questions = re.findall('utterance "(.*)"', s)
    targets = re.findall('targetFormula \(string "(.*)"', s)
    targets = [normalize_target(t) for t in targets]

    pairs = [(q, t) for q, t in zip(questions, targets)]
    questions = Lang('Questions')
    targets = Lang('Targets')

    for pair in pairs:
        questions.index_words(pair[0])
        targets.index_words(pair[1])

    return questions, targets, pairs


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence, cuda=True):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if cuda:
        var = var.cuda()
    return var


def variables_from_pair(pair, lang1, lang2):
    input_variable = variable_from_sentence(lang1, pair[0])
    target_variable = variable_from_sentence(lang2, pair[1])
    return (input_variable, target_variable)


class TranslationDataset(Dataset):
    def __init__(self, lang1, lang2, pairs):
        self.lang1 = lang1
        self.lang2 = lang2
        self.pairs = pairs
        self.dataset_len = len(pairs)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.pairs[idx]
