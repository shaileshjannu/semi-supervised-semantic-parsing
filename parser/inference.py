import torch
from torch.autograd import Variable
import numpy as np

from language import SOS_token, EOS_token


class BeamSearchState:
    def __init__(self, state, score, path=[]):
        self.state = state
        self.score = score
        self.path = path


def beam_search(beam_size, decoder, decoder_hidden, encoder_outputs, targets_lang, max_length=100):
    best_path = BeamSearchState(None, -np.inf)
    initial_state = BeamSearchState(state=decoder_hidden, path=[SOS_token], score=0)
    states = [initial_state]
    current_length = 0
    while states and current_length < max_length:
        new_states = []
        for state in states:
            decoder_input = Variable(torch.LongTensor([[state.path[-1]]])).cuda()
            decoder_hidden = state.state
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topp, topv = decoder_output.topk(beam_size)
            for logp, v in zip(topp.view(-1).data.tolist(), topv.view(-1).data.tolist()):
                new_state = BeamSearchState(state=decoder_hidden, path=state.path + [v], score=state.score + logp)
                if v == EOS_token and best_path.score < new_state.score:
                    best_path = new_state
                    continue
                new_states.append(new_state)
        new_states.sort(key=lambda x: x.score, reverse=True)
        states = new_states[:beam_size]
        current_length += 1
        if not any([s for s in states if best_path.score < s.score]):
            break
    return ' '.join([targets_lang.index2word[ni] for ni in best_path.path][1:-1])