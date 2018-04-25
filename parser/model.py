import os
import random
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from progressbar import Percentage, Bar, ETA, ProgressBar
from tensorboardX import SummaryWriter

from common import fix_parentheses
from inference import beam_search
from encoder import Encoder
from decoder import Decoder
from language import *
from preprocess import TranslationDataset, variables_from_pair, variable_from_sentence


class Model:
    def __init__(self, lang1, lang2, hidden_size, n_layers, dropout, lr, log_path,
                 teacher_forcing_ratio=0.5, clip=5.0, cuda=True):
        self.lang1 = lang1
        self.lang2 = lang2
        self.n_layers = n_layers
        self.log_path = log_path
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.clip = clip
        self.cuda = cuda
        self.best_acc = -1

        self.encoder = Encoder(lang1.n_words, hidden_size, n_layers)
        self.decoder = Decoder(hidden_size, lang2.n_words, n_layers, dropout_p=dropout)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()

        if cuda:
            self.encoder.cuda()
            self.decoder.cuda()

        os.makedirs(log_path, exist_ok=True)
        self.board_writer = SummaryWriter(os.path.join(log_path))

    def _get_next_input(self, decoder_output):
        # Choose top word from output
        _, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if self.cuda:
            decoder_input = decoder_input.cuda()
        return decoder_input, ni

    def _hidden_encoder_to_decoder(self, encoder_hidden):
        return encoder_hidden[0].view(self.n_layers, 1, -1), encoder_hidden[1].view(self.n_layers, 1, -1)

    def _train_sample(self, input_variable, target_variable):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0

        target_length = target_variable.size()[0]

        # Run words through encoder
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        # Use last hidden state from encoder to start decoder
        decoder_hidden = self._hidden_encoder_to_decoder(encoder_hidden)
        if self.cuda:
            decoder_input = decoder_input.cuda()

        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                             decoder_hidden,
                                                                             encoder_outputs)
            loss += self.criterion(torch.unsqueeze(decoder_output[0].view(-1), 0), target_variable[di])
            if use_teacher_forcing:
                decoder_input = target_variable[di]  # Next target is next input
            else:
                decoder_input, ni = self._get_next_input(decoder_output)
                if ni == EOS_token:
                    break

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / target_length

    def train(self, train_pairs, val_pairs, n_epochs, publish_freq=50, eval_freq=50):
        num_samples = len(train_pairs)
        train_dataset = TranslationDataset(self.lang1, self.lang2, train_pairs)
        train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True)

        iters = 0

        for epoch in range(n_epochs):
            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=num_samples, widgets=widgets)
            pbar.start()
            loss_to_publish = 0

            for i, sample in enumerate(train_dataset):
                pbar.update(i)

                sample = variables_from_pair((sample[0][0], sample[1][0]), self.lang1, self.lang2)
                input_variable = sample[0]
                target_variable = sample[1]

                # Run the train function
                loss_to_publish += self._train_sample(input_variable, target_variable)
                iters += 1

                if iters % publish_freq == 0:
                    loss_to_publish /= publish_freq
                    self.board_writer.add_scalar('Training loss', loss_to_publish, iters)

                if iters % eval_freq == 0:
                    acc, preds = self.evaluate(val_pairs)
                    self.board_writer.add_scalar('Val acc', acc, iters)
                    predictions = ['{0}\t{1}\t{2}\n'.format(q, t, i) for i, (q, t) in zip(preds, val_pairs)]
                    output_path = os.path.join(self.log_path, 'val_results_{0}.tsv'.format(iters))
                    with open(output_path, 'w') as output:
                        output.writelines(predictions)
                    if acc > self.best_acc:
                        torch.save(self.encoder, os.path.join(self.log_path, 'encoder'))
                        torch.save(self.decoder, os.path.join(self.log_path, 'decoder'))
                        self.best_acc = acc

    def predict(self, input):
        input_variable = variable_from_sentence(self.lang1, input)

        # Run through encoder
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_hidden = self._hidden_encoder_to_decoder(encoder_hidden)

        query = beam_search(5, self.decoder, decoder_hidden, encoder_outputs, self.lang2)
        query = fix_parentheses(query)
        return query

    def evaluate(self, pairs):
        corrects = 0
        preds = []
        for input, target in pairs:
            pred = self.predict(input)
            preds.append(pred)
            if pred == target:
                corrects += 1
        return corrects / len(pairs), preds
