import torch
import torch.nn as nn
from config import Config


class Generator(nn.Module):
    def __init__(self, config: Config):
        super(Generator, self).__init__()
        self.image_feature_channels = config.image_feature_channels
        self.image_feature_dim = config.image_feature_dim
        self.word_embedding_dim = config.word_embedding_dim
        self.vocabulary_size = config.vocabulary_size
        self.decoder_dim = config.decoder_dim
        self.attention_dim = config.attention_dim
        self.max_sentence_length = config.max_sentence_length
        self.beam_size = config.beam_size
        self.dropout_rate = config.dropout_rate
        self.device = torch.device('cuda')

    def initialize(self, embedding_vectors):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
            elif isinstance(layer, nn.LSTMCell):
                for parameter in layer.parameters():
                    if len(parameter.size()) >= 2:
                        nn.init.orthogonal_(parameter.data)
                    else:
                        nn.init.zeros_(parameter.data)
        self.W_p.bias.data.fill_(0)
        self.W_p.weight.data.uniform_(-0.1, 0.1)
        assert self.vocabulary_size == embedding_vectors.size(0), 'embedding vector dimension error'
        self.embedding_layer.weight.data.copy_(embedding_vectors)

class BeamWord():
    def __init__(self, y, h, m, previous_word, log_prob):
        self.y = y
        self.h = h
        self.m = m
        self.previous_word = previous_word
        self.log_prob = log_prob

    def __lt__(self, other):
        return self.log_prob > other.log_prob

    def __gt__(self, other):
        return self.log_prob < other.log_prob
