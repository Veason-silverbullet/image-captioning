import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from util import Config


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

class ShowTell(nn.Module):
    def __init__(self, config: Config):
        super(ShowTell, self).__init__()
        self.model_name = 'show_tell'
        self.cnn_encoder = config.cnn_encoder.lower()
        self.word_embedding_dim = config.word_embedding_dim
        self.vocabulary_size = config.vocabulary_size
        self.decoder_dim = config.decoder_dim
        self.max_sentence_length = config.max_sentence_length
        self.beam_size = config.beam_size
        self.dropout_rate = config.dropout_rate
        self.device = torch.device('cuda')

        if self.cnn_encoder == 'resnet50' or self.cnn_encoder == 'resnet-50':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet101' or self.cnn_encoder == 'resnet-101':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet152' or self.cnn_encoder == 'resnet-152':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet152(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'vgg19' or self.cnn_encoder == 'vgg-19':
            self.image_feature_channels = 512
            self.image_feature_dim = 14 * 14
            self.encoder = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).children())[:-2][0][:-2])
        elif self.cnn_encoder == 'densenet121' or self.cnn_encoder == 'densenet-121':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet121(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet161' or self.cnn_encoder == 'densenet-161':
            self.image_feature_channels = 2208
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet161(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet169' or self.cnn_encoder == 'densenet-169':
            self.image_feature_channels = 1664
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet169(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'googlenet' or self.cnn_encoder == 'inception-v1':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.googlenet(pretrained=True).children())[:-3])
        else:
            raise Exception('cnn encoder type not support: %s', cnn_encoder)

        self.h_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.m_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.lstm_decoder = nn.LSTMCell(input_size=self.word_embedding_dim, hidden_size=self.decoder_dim, bias=True)
        self.W_p = nn.Linear(in_features=self.decoder_dim, out_features=self.vocabulary_size, bias=True)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.word_embedding_dim)
        self.decode_y_cpu = torch.zeros([self.beam_size], dtype=torch.long, device=torch.device('cpu'))                               # [beam_size]
        self.decode_y = torch.zeros([self.beam_size], dtype=torch.long, device=self.device)                                           # [beam_size]
        self.decode_h = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                           # [beam_size, decoder_dim]
        self.decode_m = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                           # [beam_size, decoder_dim]
        self.decode_zero_embedding = torch.zeros([1, self.word_embedding_dim], device=self.device)                                    # [1, word_embedding_dim]

        for param in self.encoder.parameters():
            param.requires_grad = config.finetune_encoder if config.mode == 'train' else False
        self.embedding_layer.requires_grad = config.finetune_word_embedding if config.mode == 'train' else False

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

    def forward(self, images, target, max_step=None):
        batch_size = images.size(0)
        image_feature = self.encoder(images).view([batch_size, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1) # [batch_size, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                          # [batch_size, channels]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                     # [batch_size, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                     # [batch_size, decoder_dim]
        logits = torch.zeros([batch_size, self.max_sentence_length, self.vocabulary_size], device=self.device)                        # [batch_size, max_sentence_length, vocabulary_size]
        word_embeddings = self.embedding_layer(target)                                                                                # [batch_size, max_sentence_length, word_embedding_dim]
        if max_step is None:
            max_step = self.max_sentence_length

        for i in range(max_step):
            if i == 0:
                word_embedding = torch.zeros([batch_size, self.word_embedding_dim], device=self.device)                               # [batch_size, word_embedding_dim]
            else:
                word_embedding = word_embeddings[:, i - 1]                                                                            # [batch_size, word_embedding_dim]
            h, m = self.lstm_decoder(word_embedding, (h, m))                                                                          # [batch_size, decoder_dim]
            logits[:, i, :] = self.W_p(F.dropout(h, p=self.dropout_rate, inplace=True, training=self.training))                       # [batch_size, max_sentence_length, vocabulary_size]

        return F.log_softmax(logits, dim=2)

    def decode(self, images):
        image_feature = self.encoder(images).view([1, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1)          # [1, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                          # [1, channels]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                     # [1, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                     # [1, decoder_dim]
        h, m = self.lstm_decoder(self.decode_zero_embedding, (h, m))                                                                  # [1, decoder_dim]
        logits = F.log_softmax(self.W_p(h), dim=1)                                                                                    # [1, vocabulary_size]
        log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                              # [1, beam_size]
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
            self.decode_y.copy_(self.decode_y_cpu)                                                                                    # [beam_size]
            word_embedding = self.embedding_layer(self.decode_y)                                                                      # [beam_size, word_embedding_dim]
            h, m = self.lstm_decoder(word_embedding, (self.decode_h, self.decode_m))                                                  # [beam_size, decoder_dim]
            logits = F.log_softmax(self.W_p(h), dim=1)                                                                                # [beam_size, vocabulary_size]
            log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                          # [beam_size, beam_size]
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

class ShowAttendTell(nn.Module):
    def __init__(self, config: Config):
        super(ShowAttendTell, self).__init__()
        self.model_name = 'show_attend_tell'
        self.cnn_encoder = config.cnn_encoder.lower()
        self.word_embedding_dim = config.word_embedding_dim
        self.vocabulary_size = config.vocabulary_size
        self.decoder_dim = config.decoder_dim
        self.max_sentence_length = config.max_sentence_length
        self.beam_size = config.beam_size
        self.dropout_rate = config.dropout_rate
        self.device = torch.device('cuda')

        if self.cnn_encoder == 'resnet50' or self.cnn_encoder == 'resnet-50':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet101' or self.cnn_encoder == 'resnet-101':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet152' or self.cnn_encoder == 'resnet-152':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet152(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'vgg19' or self.cnn_encoder == 'vgg-19':
            self.image_feature_channels = 512
            self.image_feature_dim = 14 * 14
            self.encoder = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).children())[:-2][0][:-2])
        elif self.cnn_encoder == 'densenet121' or self.cnn_encoder == 'densenet-121':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet121(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet161' or self.cnn_encoder == 'densenet-161':
            self.image_feature_channels = 2208
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet161(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet169' or self.cnn_encoder == 'densenet-169':
            self.image_feature_channels = 1664
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet169(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'googlenet' or self.cnn_encoder == 'inception-v1':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.googlenet(pretrained=True).children())[:-3])
        else:
            raise Exception('cnn encoder type not support: %s', cnn_encoder)

        self.h_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.m_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.lstm_decoder = nn.LSTMCell(input_size=2*self.decoder_dim+self.word_embedding_dim, hidden_size=self.decoder_dim, bias=True)
        self.W_a = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.W_b = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.W_v = nn.Linear(in_features=self.decoder_dim, out_features=self.image_feature_dim, bias=False)
        self.W_g = nn.Linear(in_features=self.decoder_dim, out_features=self.image_feature_dim, bias=False)
        self.w_h = nn.Linear(in_features=self.image_feature_dim, out_features=1, bias=False)
        self.fc = nn.Linear(in_features=self.decoder_dim, out_features=self.decoder_dim, bias=True)
        self.W_p = nn.Linear(in_features=self.decoder_dim, out_features=self.vocabulary_size, bias=True)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.word_embedding_dim)
        self.decode_y_cpu = torch.zeros([self.beam_size], dtype=torch.long, device=torch.device('cpu'))                               # [beam_size]
        self.decode_y = torch.zeros([self.beam_size], dtype=torch.long, device=self.device)                                           # [beam_size]
        self.decode_h = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                           # [beam_size, decoder_dim]
        self.decode_m = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                           # [beam_size, decoder_dim]
        self.decode_zero_embedding = torch.zeros([1, self.word_embedding_dim], device=self.device)                                    # [1, word_embedding_dim]

        for param in self.encoder.parameters():
            param.requires_grad = config.finetune_encoder if config.mode == 'train' else False
        self.embedding_layer.requires_grad = config.finetune_word_embedding if config.mode == 'train' else False

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

    # Input  : visual feature and LSTM hidden state
    # V      : [batch_size, image_feature_dim, decoder_dim]
    # h      : [batch_size, decoder_dim]
    # Output : contextual visual feature and attention weight
    # c      : [batch_size, decoder_dim]
    # alpha  : [batch_size, image_feature_dim]
    def attention(self, V, h):
        z = self.w_h(torch.tanh(self.W_v(V) + self.W_g(h).unsqueeze(dim=1))).squeeze(dim=2)                                           # [batch_size, image_feature_dim]
        alpha = F.softmax(z, dim=1)                                                                                                   # [batch_size, image_feature_dim]
        c = torch.bmm(alpha.unsqueeze(dim=1), V).squeeze(dim=1)                                                                       # [batch_size, decoder_dim]
        return c, alpha

    def forward(self, images, target, max_step=None):
        batch_size = images.size(0)
        image_feature = self.encoder(images).view([batch_size, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1) # [batch_size, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                          # [batch_size, channels]
        V = F.dropout(F.relu(self.W_a(image_feature), inplace=True), p=self.dropout_rate, training=self.training)                     # [batch_size, image_feature_dim, decoder_dim]
        v_g = F.dropout(F.relu(self.W_b(mean_image_feature), inplace=True), p=self.dropout_rate, training=self.training)              # [batch_size, decoder_dim]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                     # [batch_size, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                     # [batch_size, decoder_dim]
        logits = torch.zeros([batch_size, self.max_sentence_length, self.vocabulary_size], device=self.device)                        # [batch_size, max_sentence_length, vocabulary_size]
        attention_weights = torch.zeros([batch_size, self.max_sentence_length, self.image_feature_dim], device=self.device)           # [batch_size, max_sentence_length, image_feature_dim]
        word_embeddings = self.embedding_layer(target)                                                                                # [batch_size, max_sentence_length, word_embedding_dim]
        if max_step is None:
            max_step = self.max_sentence_length

        for i in range(max_step):
            if i == 0:
                word_embedding = torch.zeros([batch_size, self.word_embedding_dim], device=self.device)                               # [batch_size, word_embedding_dim]
            else:
                word_embedding = word_embeddings[:, i - 1]                                                                            # [batch_size, word_embedding_dim]
            image_context, attention_weight = self.attention(V, h)                                                                    # [batch_size, decoder_dim]
            h, m = self.lstm_decoder(torch.cat([image_context, v_g, word_embedding], dim=1), (h, m))                                  # [batch_size, decoder_dim]
            out = F.dropout(torch.tanh(self.fc(h + image_context)), p=self.dropout_rate, training=self.training)                      # [batch_size, decoder_dim]
            logits[:, i, :] = self.W_p(out)                                                                                           # [batch_size, max_sentence_length, vocabulary_size]
            attention_weights[:, i, :] = attention_weight                                                                             # [batch_size, max_sentence_length, image_feature_dim]

        return F.log_softmax(logits, dim=2), attention_weights

    def decode(self, images):
        image_feature = self.encoder(images).view([1, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1)          # [1, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                          # [1, channels]
        V = F.relu(self.W_a(image_feature), inplace=True)                                                                             # [1, image_feature_dim, decoder_dim]
        v_g = F.relu(self.W_b(mean_image_feature), inplace=True)                                                                      # [1, decoder_dim]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                     # [1, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                     # [1, decoder_dim]
        image_context, attention_weight = self.attention(V, h)                                                                        # [1, decoder_dim]
        h, m = self.lstm_decoder(torch.cat([image_context, v_g, self.decode_zero_embedding], dim=1), (h, m))                          # [1, decoder_dim]
        out = torch.tanh(self.fc(h + image_context))                                                                                  # [1, decoder_dim]
        logits = F.log_softmax(self.W_p(out), dim=1)                                                                                  # [1, vocabulary_size]
        log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                              # [1, beam_size]
        log_probs = log_probs.tolist()
        indices = indices.tolist()
        V = V.expand(self.beam_size, -1, -1)                                                                                          # [beam_size, channels]
        v_g = v_g.expand(self.beam_size, -1)                                                                                          # [beam_size, image_feature_dim, decoder_dim]

        beam_word_list1 = [BeamWord(indices[0][i], h, m, None, log_probs[0][i]) for i in range(self.beam_size)]
        beam_word_list2 = []
        best_prob = None
        best_hypothesis = None

        for step in range(self.max_sentence_length - 1):
            for i, beam_word in enumerate(beam_word_list1):
                self.decode_y_cpu[i] = beam_word.y
                self.decode_h[i] = beam_word.h
                self.decode_m[i] = beam_word.m
            self.decode_y.copy_(self.decode_y_cpu)                                                                                    # [beam_size]
            word_embedding = self.embedding_layer(self.decode_y)                                                                      # [beam_size, word_embedding_dim]
            image_context, attention_weight = self.attention(V, self.decode_h)                                                        # [beam_size, decoder_dim]
            h, m = self.lstm_decoder(torch.cat([image_context, v_g, word_embedding], dim=1), (self.decode_h, self.decode_m))          # [beam_size, decoder_dim]
            out = torch.tanh(self.fc(h + image_context))                                                                              # [beam_size, decoder_dim]
            logits = F.log_softmax(self.W_p(out), dim=1)                                                                              # [beam_size, vocabulary_size]
            log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                          # [beam_size, beam_size]
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

class AdaptiveAttention(nn.Module):
    def __init__(self, config: Config):
        super(AdaptiveAttention, self).__init__()
        self.model_name = 'adaptive_attention'
        self.cnn_encoder = config.cnn_encoder.lower()
        self.word_embedding_dim = config.word_embedding_dim
        self.vocabulary_size = config.vocabulary_size
        self.decoder_dim = config.decoder_dim
        self.max_sentence_length = config.max_sentence_length
        self.beam_size = config.beam_size
        self.dropout_rate = config.dropout_rate
        self.device = torch.device('cuda')

        if self.cnn_encoder == 'resnet50' or self.cnn_encoder == 'resnet-50':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet101' or self.cnn_encoder == 'resnet-101':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'resnet152' or self.cnn_encoder == 'resnet-152':
            self.image_feature_channels = 2048
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.resnet152(pretrained=True).children())[:-2])
        elif self.cnn_encoder == 'vgg19' or self.cnn_encoder == 'vgg-19':
            self.image_feature_channels = 512
            self.image_feature_dim = 14 * 14
            self.encoder = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).children())[:-2][0][:-2])
        elif self.cnn_encoder == 'densenet121' or self.cnn_encoder == 'densenet-121':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet121(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet161' or self.cnn_encoder == 'densenet-161':
            self.image_feature_channels = 2208
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet161(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'densenet169' or self.cnn_encoder == 'densenet-169':
            self.image_feature_channels = 1664
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.densenet169(pretrained=True).children())[:-1])
        elif self.cnn_encoder == 'googlenet' or self.cnn_encoder == 'inception-v1':
            self.image_feature_channels = 1024
            self.image_feature_dim = 7 * 7
            self.encoder = nn.Sequential(*list(torchvision.models.googlenet(pretrained=True).children())[:-3])
        else:
            raise Exception('cnn encoder type not support: %s', cnn_encoder)

        self.h_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        
        self.m_init = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.lstm_decoder = nn.LSTMCell(input_size=self.decoder_dim+self.word_embedding_dim, hidden_size=self.decoder_dim, bias=True)
        self.W_a = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.W_b = nn.Linear(in_features=self.image_feature_channels, out_features=self.decoder_dim, bias=True)
        self.W_v = nn.Linear(in_features=self.decoder_dim, out_features=self.image_feature_dim, bias=False)
        self.W_g = nn.Linear(in_features=self.decoder_dim, out_features=self.image_feature_dim, bias=False)
        self.w_h = nn.Linear(in_features=self.image_feature_dim, out_features=1, bias=False)
        self.W_H = nn.Linear(in_features=self.decoder_dim, out_features=self.decoder_dim, bias=False)
        self.W_x = nn.Linear(in_features=self.decoder_dim+self.word_embedding_dim, out_features=self.decoder_dim, bias=False)
        self.W_h = nn.Linear(in_features=self.decoder_dim, out_features=self.decoder_dim, bias=False)
        self.W_s = nn.Linear(in_features=self.decoder_dim, out_features=self.image_feature_dim, bias=False)
        self.W_c = nn.Linear(in_features=self.decoder_dim, out_features=self.decoder_dim, bias=True)
        self.fc = nn.Linear(in_features=self.decoder_dim, out_features=self.decoder_dim, bias=True)
        self.W_p = nn.Linear(in_features=self.decoder_dim, out_features=self.vocabulary_size, bias=True)
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.word_embedding_dim)
        self.decode_y_cpu = torch.zeros([self.beam_size], dtype=torch.long, device=torch.device('cpu'))                               # [beam_size]
        self.decode_y = torch.zeros([self.beam_size], dtype=torch.long, device=self.device)                                           # [beam_size]
        self.decode_h = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                           # [beam_size, decoder_dim]
        self.decode_m = torch.zeros([self.beam_size, self.decoder_dim], device=self.device)                                           # [beam_size, decoder_dim]
        self.decode_zero_embedding = torch.zeros([1, self.word_embedding_dim], device=self.device)                                    # [1, word_embedding_dim]

        for param in self.encoder.parameters():
            param.requires_grad = config.finetune_encoder if config.mode == 'train' else False
        self.embedding_layer.requires_grad = config.finetune_word_embedding if config.mode == 'train' else False

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

    # Input  : visual feature, LSTM hidden state, previous LSTM hidden state, fused_feature and LSTM memory
    # V      : [batch_size, image_feature_dim, decoder_dim]
    # h      : [batch_size, decoder_dim]
    # _h     : [batch_size, decoder_dim]
    # x      : [batch_size, decoder_dim + word_embedding_dim]
    # m      : [batch_size, decoder_dim]
    # Output : contextual visual feature and attention weight
    # c      : [batch_size, decoder_dim]
    # alpha  : [batch_size, image_feature_dim]
    def adaptive_attention(self, V, h, _h, x, m):
        z = self.w_h(torch.tanh(self.W_v(V) + self.W_g(h).unsqueeze(dim=1))).squeeze(dim=2)                                           # [batch_size, image_feature_dim]
        g = torch.sigmoid(self.W_x(x) + self.W_h(_h))                                                                                 # [batch_size, decoder_dim]
        s = g * torch.tanh(m)                                                                                                         # [batch_size, decoder_dim]
        s = F.dropout(F.relu(self.W_c(s), inplace=True), p=self.dropout_rate, training=self.training)                                 # [batch_size, decoder_dim]
        alpha = F.softmax(torch.cat([z, self.w_h(torch.tanh(self.W_s(s) + self.W_g(h)))], dim=1), dim=1)                              # [batch_size, image_feature_dim + 1]                                                                                       # [batch_size, image_feature_dim]
        c = torch.bmm(alpha.unsqueeze(dim=1), torch.cat([V, s.unsqueeze(dim=1)], dim=1)).squeeze(dim=1)                               # [batch_size, decoder_dim]
        return c, alpha

    def forward(self, images, target, max_step=None):
        batch_size = images.size(0)
        image_feature = self.encoder(images).view([batch_size, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1) # [batch_size, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                          # [batch_size, channels]
        V = F.dropout(F.relu(self.W_a(image_feature), inplace=True), p=self.dropout_rate, training=self.training)                     # [batch_size, image_feature_dim, decoder_dim]
        v_g = F.dropout(F.relu(self.W_b(mean_image_feature), inplace=True), p=self.dropout_rate, training=self.training)              # [batch_size, decoder_dim]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                     # [batch_size, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                     # [batch_size, decoder_dim]
        logits = torch.zeros([batch_size, self.max_sentence_length, self.vocabulary_size], device=self.device)                        # [batch_size, max_sentence_length, vocabulary_size]
        attention_weights = torch.zeros([batch_size, self.max_sentence_length, self.image_feature_dim], device=self.device)           # [batch_size, max_sentence_length, image_feature_dim]
        word_embeddings = self.embedding_layer(target)                                                                                # [batch_size, max_sentence_length, word_embedding_dim]
        if max_step is None:
            max_step = self.max_sentence_length

        for i in range(max_step):
            if i == 0:
                word_embedding = torch.zeros([batch_size, self.word_embedding_dim], device=self.device)                               # [batch_size, word_embedding_dim]
            else:
                word_embedding = word_embeddings[:, i - 1]                                                                            # [batch_size, word_embedding_dim]
            x = torch.cat([v_g, word_embedding], dim=1)                                                                               # [batch_size, decoder_dim + word_embedding_dim]
            _h = h                                                                                                                    # [batch_size, decoder_dim]
            h, m = self.lstm_decoder(x, (h, m))                                                                                       # [batch_size, decoder_dim]
            H = F.dropout(F.relu(self.W_H(h), inplace=True), p=self.dropout_rate, training=self.training)                             # [batch_size, decoder_dim]
            image_context, attention_weight = self.adaptive_attention(V, H, _h, x, m)                                                 # [batch_size, decoder_dim]
            out = F.dropout(torch.tanh(self.fc(H + image_context)), p=self.dropout_rate, training=self.training)                      # [batch_size, decoder_dim]
            logits[:, i, :] = self.W_p(out)                                                                                           # [batch_size, max_sentence_length, vocabulary_size]
            attention_weights[:, i, :] = attention_weight[:, :self.image_feature_dim]                                                 # [batch_size, max_sentence_length, image_feature_dim]

        return F.log_softmax(logits, dim=2), attention_weights

    def decode(self, images):
        image_feature = self.encoder(images).view([1, self.image_feature_channels, self.image_feature_dim]).permute(0, 2, 1)          # [1, image_feature_dim, channels]
        mean_image_feature = torch.mean(image_feature, dim=1, keepdim=False)                                                          # [1, channels]
        V = F.relu(self.W_a(image_feature), inplace=True)                                                                             # [1, image_feature_dim, decoder_dim]
        v_g = F.relu(self.W_b(mean_image_feature), inplace=True)                                                                      # [1, decoder_dim]
        h = F.relu(self.h_init(mean_image_feature), inplace=True)                                                                     # [1, decoder_dim]
        m = F.relu(self.m_init(mean_image_feature), inplace=True)                                                                     # [1, decoder_dim]
        x = torch.cat([v_g, self.decode_zero_embedding], dim=1)                                                                       # [1, decoder_dim + word_embedding_dim]
        _h = h                                                                                                                        # [1, decoder_dim]
        h, m = self.lstm_decoder(x, (h, m))                                                                                           # [1, decoder_dim]
        H = F.relu(self.W_H(h), inplace=True)                                                                                         # [1, decoder_dim]
        image_context, attention_weight = self.adaptive_attention(V, H, _h, x, m)                                                     # [1, decoder_dim]
        out = torch.tanh(self.fc(H + image_context))                                                                                  # [1, decoder_dim]
        logits = F.log_softmax(self.W_p(out), dim=1)                                                                                  # [1, vocabulary_size]
        log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                              # [1, beam_size]
        log_probs = log_probs.tolist()
        indices = indices.tolist()
        V = V.expand(self.beam_size, -1, -1)                                                                                          # [beam_size, channels]
        v_g = v_g.expand(self.beam_size, -1)                                                                                          # [beam_size, image_feature_dim, decoder_dim]

        beam_word_list1 = [BeamWord(indices[0][i], h, m, None, log_probs[0][i]) for i in range(self.beam_size)]
        beam_word_list2 = []
        best_prob = None
        best_hypothesis = None

        for step in range(self.max_sentence_length - 1):
            for i, beam_word in enumerate(beam_word_list1):
                self.decode_y_cpu[i] = beam_word.y
                self.decode_h[i] = beam_word.h
                self.decode_m[i] = beam_word.m
            self.decode_y.copy_(self.decode_y_cpu)                                                                                    # [beam_size]
            word_embedding = self.embedding_layer(self.decode_y)                                                                      # [beam_size, word_embedding_dim]
            x = torch.cat([v_g, word_embedding], dim=1)                                                                               # [beam_size, decoder_dim + word_embedding_dim]
            h, m = self.lstm_decoder(x, (self.decode_h, self.decode_m))                                                               # [beam_size, decoder_dim]
            H = F.relu(self.W_H(h), inplace=True)                                                                                     # [beam_size, decoder_dim]
            image_context, attention_weight = self.adaptive_attention(V, H, self.decode_h, x, m)                                      # [beam_size, decoder_dim]
            out = torch.tanh(self.fc(H + image_context))                                                                              # [beam_size, decoder_dim]
            logits = F.log_softmax(self.W_p(out), dim=1)                                                                              # [beam_size, vocabulary_size]
            log_probs, indices = torch.topk(logits, k=self.beam_size, dim=1)                                                          # [beam_size, beam_size]
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
