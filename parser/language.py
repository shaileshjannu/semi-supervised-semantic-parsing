SOS_token = 0
EOS_token = 1
UNK_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS,EOS and UNK

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 0

        self.word2count[word] += 1

        if self.word2count[word] == 2:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
