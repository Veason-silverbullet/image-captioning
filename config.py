import argparse
import time
import torch
import torchvision.transforms as transforms
import random
import numpy as np


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Image caption')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'decode', 'watch', 'debug'], help='mode')
        parser.add_argument('--device_id', type=int, default=0, help='specific GPU to run the model')
        parser.add_argument('--iteration_to_show', type=int, default=200, help='iteration to show the accuracy and loss')
        parser.add_argument('--seed', type=int, default=0, help='seed for random generator')
        parser.add_argument('--decode_model_path', type=str, default='', help='decode model path')
        # Dataset config
        parser.add_argument('--dataset_root', type=str, default='../../../../dataset/coco2014', help='mscoco2014 dataset root')
        parser.add_argument('--preprocess_mode', type=str, default='standard', choices=['standard', 'default', 'nltk'], help='dataset preprocessing mode')
        parser.add_argument('--word_frequency_threshold', type=int, default=5, help='the word frequency threshold for building vocabulary')
        # Model config
        parser.add_argument('--cnn_encoder', type=str, default='resnet152', help='CNN encoder backbone')
        parser.add_argument('--finetune_encoder', type=bool, default=False, help='whether finetune CNN encoder')
        parser.add_argument('--decoder_dim', type=int, default=512, help='the dimension of decoder lstm')
        parser.add_argument('--attention_dim', type=int, default=512, help='the dimension of decoder lstm')
        parser.add_argument('--word_embedding_dim', type=int, default=300, help='word embedding dimension')
        # Training config
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--epoch', type=int, default=50, help='training epoch')
        parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        parser.add_argument('--lr_decay_epoch', type=int, default=3, help='epoch of learning rate decay')
        parser.add_argument('--lr_decay', type=float, default=0.85, help='learning rate decay rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        parser.add_argument('--gradient_clip', type=float, default=5.0, help='gradient clipping value')
        parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
        parser.add_argument('--average_sentence_loss', type=bool, default=True, help='whether normalize loss by sentence length')
        parser.add_argument('--hard_attention_entropy', type=float, default=0.002, help='hard attention entropy')
        parser.add_argument('--coverage', type=bool, default=False, help='whether use coverage loss')
        parser.add_argument('--Lambda', type=float, default=1, help='Lambda for joint loss')
        parser.add_argument('--finetune_cnn_after', type=int, default=-1, help='after what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
        parser.add_argument('--cnn_learning_rate', type=float, default=1e-5, help='learning rate of fintuning CNN')
        parser.add_argument('--early_stopping_epoch', type=int, default=8, help='early stopping epoch for CIDEr')
        # Reinforcement learning config
        parser.add_argument('--RL_pretrained_model', type=str, default='show_attend_tell', choices=['show_tell', 'show_attend_tell', 'adaptive_attention'], help='reinforcement learning pretrained model')
        parser.add_argument('--gradient_estimator', type=str, default='Gumbel_softmax', choices=['REINFORCE', 'Gumbel_softmax'], help='gradient estimator of categorical sampling')
        # Decode configs
        parser.add_argument('--beam_size', type=int, default=5, help='decode beam size')

        args = parser.parse_args()
        self.mode = args.mode
        self.device_id = args.device_id
        self.iteration_to_show = args.iteration_to_show
        self.seed = args.seed
        self.decode_model_path = args.decode_model_path
        self.dataset_root = args.dataset_root
        self.preprocess_mode = args.preprocess_mode
        self.word_frequency_threshold = args.word_frequency_threshold
        self.cnn_encoder = args.cnn_encoder
        self.finetune_encoder = args.finetune_encoder
        self.decoder_dim = args.decoder_dim
        self.attention_dim = args.attention_dim
        self.word_embedding_dim = args.word_embedding_dim
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.lr = args.lr
        self.lr_decay_epoch = args.lr_decay_epoch
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.gradient_clip = args.gradient_clip
        self.dropout_rate = args.dropout_rate
        self.average_sentence_loss = args.average_sentence_loss
        self.hard_attention_entropy = args.hard_attention_entropy
        self.coverage = args.coverage
        self.Lambda = args.Lambda
        self.finetune_cnn_after = args.finetune_cnn_after
        self.cnn_learning_rate = args.cnn_learning_rate
        self.early_stopping_epoch = args.early_stopping_epoch
        self.RL_pretrained_model = args.RL_pretrained_model
        self.gradient_estimator = args.gradient_estimator
        self.beam_size = args.beam_size
        self.transforms = transforms.Compose(transforms=[transforms.Resize((224, 224)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed if self.seed >= 0 else (int)(time.time()))
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        random.seed(self.seed)
        np.random.seed(self.seed)
