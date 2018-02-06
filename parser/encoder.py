import torch
from torch import nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // 2
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, n_layers, bidirectional=True)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, cuda=True):
        num_hiddens = self.n_layers * 2
        hidden = Variable(torch.randn(num_hiddens, 1, self.hidden_size))
        context = Variable(torch.randn(num_hiddens, 1, self.hidden_size))
        if cuda:
            hidden = hidden.cuda()
            context = context.cuda()
        return (hidden, context)
