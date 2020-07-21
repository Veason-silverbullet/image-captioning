# Model: Seq2seq image caption generator
# Paper: Show and Tell: A Neural Image Caption Generator (https://arxiv.org/pdf/1411.4555.pdf)
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config
from language_generator import Generator
from language_generator import BeamWord


class ShowTell(Generator):
    def __init__(self, config: Config):
        super(ShowTell, self).__init__(config)
        self.model_name = 'show_tell'
        self.h_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim)
        self.m_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim)
        self.lstm_decoder = nn.LSTMCell(input_size=self.word_embedding_dim, hidden_size=self.decoder_dim)
        self.W_p = nn.Linear(in_features=self.decoder_dim, out_features=self.vocabulary_size)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.word_embedding_dim)
        self.decode_y_cpu = torch.zeros([self.beam_size], dtype=torch.long, device=torch.device('cpu'))         # [beam_size]
        self.decode_y = torch.zeros([self.beam_size], dtype=torch.long, device=self.device)                     # [beam_size]
        self.decode_h = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                     # [beam_size, decoder_dim]
        self.decode_m = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                     # [beam_size, decoder_dim]
        self.decode_zero_embedding = torch.zeros([1, self.word_embedding_dim], device=self.device)              # [1, word_embedding_dim]

    def forward(self, image_feature, mean_image_feature, target, max_step=None):
        batch_size = mean_image_feature.size(0)
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                               # [batch_size, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                               # [batch_size, decoder_dim]
        logits = torch.zeros([batch_size, self.max_sentence_length, self.vocabulary_size], device=self.device)  # [batch_size, max_sentence_length, vocabulary_size]
        word_embeddings = F.dropout(self.embedding_layer(target), p=self.dropout_rate, training=self.training)  # [batch_size, max_sentence_length, word_embedding_dim]
        if max_step is None:
            max_step = self.max_sentence_length

        for i in range(max_step):
            if i == 0:
                word_embedding = torch.zeros([batch_size, self.word_embedding_dim], device=self.device)         # [batch_size, word_embedding_dim]
            else:
                word_embedding = word_embeddings[:, i - 1]                                                      # [batch_size, word_embedding_dim]
            h, m = self.lstm_decoder(word_embedding, (h, m))                                                    # [batch_size, decoder_dim]
            logits[:, i, :] = self.W_p(F.dropout(h, p=self.dropout_rate, inplace=True, training=self.training)) # [batch_size, max_sentence_length, vocabulary_size]

        return F.log_softmax(logits, dim=2)

    def decode(self, image_feature, mean_image_feature):
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                               # [1, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                               # [1, decoder_dim]
        h, m = self.lstm_decoder(self.decode_zero_embedding, (h, m))                                            # [1, decoder_dim]
        logits = F.log_softmax(self.W_p(h), dim=1)                                                              # [1, vocabulary_size]
        log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                        # [1, beam_size]
        log_probs = log_probs.tolist()
        indices = indices.tolist()

        beam_word_list1 = [BeamWord(indices[0][i], h, m, None, log_probs[0][i]) for i in range(self.beam_size)]
        beam_word_list2 = []
        best_prob = None
        best_hypothesis = None

        for step in range(self.max_sentence_length - 1):
            for i, beam_word in enumerate(beam_word_list1):
                self.decode_y_cpu[i] = beam_word.y
                self.decode_h[i] = beam_word.h
                self.decode_m[i] = beam_word.m
            self.decode_y.copy_(self.decode_y_cpu)                                                              # [beam_size]
            word_embedding = self.embedding_layer(self.decode_y)                                                # [beam_size, word_embedding_dim]
            h, m = self.lstm_decoder(word_embedding, (self.decode_h, self.decode_m))                            # [beam_size, decoder_dim]
            logits = F.log_softmax(self.W_p(h), dim=1)                                                          # [beam_size, vocabulary_size]
            log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                    # [beam_size, beam_size]
            log_probs = log_probs.tolist()
            indices = indices.tolist()

            for i, beam_word in enumerate(beam_word_list1):
                for j in range(self.beam_size):
                    y = indices[i][j]
                    log_prob = beam_word.log_prob + log_probs[i][j]
                    if y == 0:
                        if best_hypothesis is None or log_prob > best_prob:
                            best_hypothesis = BeamWord(0, None, None, beam_word, log_prob)
                            best_prob = log_prob
                    elif best_hypothesis is None or log_prob > best_prob:
                        beam_word_list2.append(BeamWord(y, h[i], m[i], beam_word, log_prob))

            if len(beam_word_list2) == 0:
                break
            beam_word_list2.sort()
            beam_word_list1 = [beam_word_list2[i] for i in range(min(self.beam_size, len(beam_word_list2)))]
            beam_word_list2.clear()

        if best_hypothesis is None:
            best_hypothesis = beam_word_list1[0]
        hypothesis = []
        if best_hypothesis.y == 0:
            best_hypothesis = best_hypothesis.previous_word
        while True:
            hypothesis.append(best_hypothesis.y)
            if best_hypothesis.previous_word != None:
                best_hypothesis = best_hypothesis.previous_word
            else:
                break
        hypothesis.reverse()
        return hypothesis
