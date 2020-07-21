# Model: Seq2seq-attention image caption generator
# Paper: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (https://arxiv.org/abs/1502.03044)
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import Config
from language_generator import Generator
from language_generator import BeamWord
from torch.distributions import Categorical


# input  : probs           # [batch_size, class_num]
#        : features        # [batch_size, class_num, feature_dim]
# output : sample_features # [batch_size, feature_dim]
class Gumbel_Softmax_Sampler():
    class_num = 0
    temperature = 1
    @staticmethod
    def apply(logits, features):
        sample_logits = logits - torch.log(-torch.log(torch.rand(logits.size()).cuda() + 1e-20) + 1e-20)
        sample_probs = F.softmax(sample_logits / Gumbel_Softmax_Sampler.temperature, dim=1)
        _, sample_indices = sample_logits.max(dim=1, keepdim=True)
        sample_one_hots = torch.zeros([features.size(0), Gumbel_Softmax_Sampler.class_num]).cuda().scatter_(1, sample_indices, 1)
        weights = (sample_one_hots - sample_probs).detach() + sample_probs
        return torch.bmm(weights.unsqueeze(dim=1), features).squeeze(dim=1)

# input  : probs           # [batch_size, class_num]
#        : features        # [batch_size, class_num, feature_dim]
# output : sample_features # [batch_size, feature_dim]
class REINFORCE_Sampler(torch.autograd.Function):
    class_num = 0
    feature_dim = 0
    hard_attention_entropy = 0
    baseline = 0
    loss_logprobs = None
    @staticmethod
    def forward(ctx, probs, features):
        sample_indices = Categorical(probs=probs.detach()).sample().unsqueeze(1)
        sample_probs = probs.gather(1, sample_indices)
        weights = torch.zeros([features.size(0), REINFORCE_Sampler.class_num]).cuda().scatter_(1, sample_indices, 1)
        ctx.save_for_backward(probs, sample_probs, weights)
        return torch.bmm(weights.unsqueeze(dim=1), features).squeeze(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        probs, sample_probs, weights = ctx.saved_tensors
        grad_input_1 = sample_probs * (grad_output.sum(dim=1, keepdim=True) * weights + 0.1 * REINFORCE_Sampler.loss_logprobs / probs) \
                     + REINFORCE_Sampler.hard_attention_entropy * (torch.log(probs) + 1)
        grad_input_2 = torch.bmm(weights.unsqueeze(dim=2), grad_output.unsqueeze(dim=1))
        return grad_input_1, grad_input_2


class HardAttention(Generator):
    def __init__(self, config: Config):
        super(HardAttention, self).__init__(config)
        self.model_name = 'hard_attention-' + config.gradient_estimator
        self.lstm_decoder = nn.LSTMCell(input_size=2*self.decoder_dim+self.word_embedding_dim, hidden_size=self.decoder_dim)
        self.h_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim)
        self.m_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim)
        self.W_a = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim)
        self.W_b = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim)
        self.W_v = nn.Linear(in_features=self.decoder_dim, out_features=self.attention_dim)
        self.W_g = nn.Linear(in_features=self.decoder_dim, out_features=self.attention_dim)
        self.w_h = nn.Linear(in_features=self.attention_dim, out_features=1)
        self.fc = nn.Linear(in_features=self.decoder_dim, out_features=self.decoder_dim)
        self.W_p = nn.Linear(in_features=self.decoder_dim, out_features=self.vocabulary_size)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.word_embedding_dim)
        self.decode_y_cpu = torch.zeros([self.beam_size], dtype=torch.long, device=torch.device('cpu'))                            # [beam_size]
        self.decode_y = torch.zeros([self.beam_size], dtype=torch.long, device=self.device)                                        # [beam_size]
        self.decode_h = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                        # [beam_size, decoder_dim]
        self.decode_m = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                        # [beam_size, decoder_dim]
        self.decode_zero_embedding = torch.zeros([1, self.word_embedding_dim], device=self.device)                                 # [1, word_embedding_dim]
        self.gradient_estimator = config.gradient_estimator
        if config.gradient_estimator == 'REINFORCE':
            self.sampler = REINFORCE_Sampler.apply
            REINFORCE_Sampler.class_num = self.image_feature_dim
            REINFORCE_Sampler.feature_dim = self.decoder_dim
            REINFORCE_Sampler.hard_attention_entropy = config.hard_attention_entropy
        else:
            self.sampler = Gumbel_Softmax_Sampler.apply
            Gumbel_Softmax_Sampler.class_num = self.image_feature_dim
            self.step = 0

    # Input       : visual feature, visual feature embedding and LSTM hidden state
    # V           : [batch_size, image_feature_dim, decoder_dim]
    # V_embedding : [batch_size, image_feature_dim, attention_dim]
    # h           : [batch_size, decoder_dim]
    # Output      : contextual visual feature and attention weight
    # c           : [batch_size, decoder_dim]
    # alpha       : [batch_size, image_feature_dim]
    def attention(self, V, V_embedding, h):
        h_embedding = self.W_g(h).unsqueeze(dim=1)                                                                                 # [batch_size, 1, attention_dim]
        z = self.w_h(F.dropout(torch.tanh(V_embedding + h_embedding), p=self.dropout_rate, training=self.training)).squeeze(dim=2) # [batch_size, image_feature_dim]
        alpha = F.softmax(z, dim=1)                                                                                                # [batch_size, image_feature_dim]
        if self.training:
            if self.gradient_estimator == 'REINFORCE':
                c = self.sampler(alpha, V)                                                                                         # [batch_size, decoder_dim]
            else:
                c = self.sampler(F.log_softmax(z, dim=1), V)                                                                       # [batch_size, decoder_dim]
        else:
            _, indices = z.max(dim=1, keepdim=True)                                                                                # [batch_size, 1]
            weights = torch.zeros([V.size(0), self.image_feature_dim]).cuda().scatter_(1, indices, 1)                              # [batch_size, image_feature_dim]
            c = torch.bmm(weights.unsqueeze(dim=1), V).squeeze(dim=1)                                                              # [batch_size, decoder_dim]
        return c, alpha

    def forward(self, image_feature, mean_image_feature, target, max_step=None):
        batch_size = image_feature.size(0)
        V = F.dropout(F.relu(self.W_a(image_feature), inplace=True), p=self.dropout_rate, training=self.training)                  # [batch_size, image_feature_dim, decoder_dim]
        v_g = F.dropout(F.relu(self.W_b(mean_image_feature), inplace=True), p=self.dropout_rate, training=self.training)           # [batch_size, decoder_dim]
        V_embedding = self.W_v(V)                                                                                                  # [batch_size, image_feature_dim, attention_dim]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                  # [batch_size, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                  # [batch_size, decoder_dim]
        logits = torch.zeros([batch_size, self.max_sentence_length, self.vocabulary_size], device=self.device)                     # [batch_size, max_sentence_length, vocabulary_size]
        attention_weights = torch.zeros([batch_size, self.max_sentence_length, self.image_feature_dim], device=self.device)        # [batch_size, max_sentence_length, image_feature_dim]
        word_embeddings = F.dropout(self.embedding_layer(target), p=self.dropout_rate, training=self.training)                     # [batch_size, max_sentence_length, word_embedding_dim]
        if max_step is None:
            max_step = self.max_sentence_length
        if self.training and self.gradient_estimator == 'Gumbel_softmax':
            Gumbel_Softmax_Sampler.temperature = max(0.4, math.exp(-self.step * 1e-5))
            self.step += 1

        for i in range(max_step):
            if i == 0:
                word_embedding = torch.zeros([batch_size, self.word_embedding_dim], device=self.device)                            # [batch_size, word_embedding_dim]
            else:
                word_embedding = word_embeddings[:, i - 1]                                                                         # [batch_size, word_embedding_dim]
            image_context, attention_weight = self.attention(V, V_embedding, h)                                                    # [batch_size, decoder_dim]
            h, m = self.lstm_decoder(torch.cat([image_context, v_g, word_embedding], dim=1), (h, m))                               # [batch_size, decoder_dim]
            out = F.dropout(torch.tanh(self.fc(h + image_context)), p=self.dropout_rate, training=self.training)                   # [batch_size, decoder_dim]
            logits[:, i, :] = self.W_p(out)                                                                                        # [batch_size, max_sentence_length, vocabulary_size]
            attention_weights[:, i, :] = attention_weight                                                                          # [batch_size, max_sentence_length, image_feature_dim]

        return F.log_softmax(logits, dim=2), attention_weights

    def decode(self, image_feature, mean_image_feature):
        V = F.relu(self.W_a(image_feature), inplace=True)                                                                          # [1, image_feature_dim, decoder_dim]
        v_g = F.relu(self.W_b(mean_image_feature), inplace=True)                                                                   # [1, decoder_dim]
        V_embedding = self.W_v(V)                                                                                                  # [1, image_feature_dim, attention_dim]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                  # [1, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                  # [1, decoder_dim]
        image_context, attention_weight = self.attention(V, V_embedding, h)                                                        # [1, decoder_dim]
        h, m = self.lstm_decoder(torch.cat([image_context, v_g, self.decode_zero_embedding], dim=1), (h, m))                       # [1, decoder_dim]
        out = torch.tanh(self.fc(h + image_context))                                                                               # [1, decoder_dim]
        logits = F.log_softmax(self.W_p(out), dim=1)                                                                               # [1, vocabulary_size]
        log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                           # [1, beam_size]
        log_probs = log_probs.tolist()
        indices = indices.tolist()
        V = V.expand(self.beam_size, -1, -1)                                                                                       # [beam_size, image_feature_dim, decoder_dim]
        v_g = v_g.expand(self.beam_size, -1)                                                                                       # [beam_size, channels]
        V_embedding = V_embedding.expand(self.beam_size, -1, -1)                                                                   # [1, image_feature_dim, attention_dim]

        beam_word_list1 = [BeamWord(indices[0][i], h, m, None, log_probs[0][i]) for i in range(self.beam_size)]
        beam_word_list2 = []
        best_prob = None
        best_hypothesis = None

        for step in range(self.max_sentence_length - 1):
            for i, beam_word in enumerate(beam_word_list1):
                self.decode_y_cpu[i] = beam_word.y
                self.decode_h[i] = beam_word.h
                self.decode_m[i] = beam_word.m
            self.decode_y.copy_(self.decode_y_cpu)                                                                                 # [beam_size]
            word_embedding = self.embedding_layer(self.decode_y)                                                                   # [beam_size, word_embedding_dim]
            image_context, attention_weight = self.attention(V, V_embedding, self.decode_h)                                        # [beam_size, decoder_dim]
            h, m = self.lstm_decoder(torch.cat([image_context, v_g, word_embedding], dim=1), (self.decode_h, self.decode_m))       # [beam_size, decoder_dim]
            out = torch.tanh(self.fc(h + image_context))                                                                           # [beam_size, decoder_dim]
            logits = F.log_softmax(self.W_p(out), dim=1)                                                                           # [beam_size, vocabulary_size]
            log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                       # [beam_size, beam_size]
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
