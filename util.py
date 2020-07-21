import os
from config import Config
from _coco import _CocoCaptions
import torch
from torch.distributions import Categorical


def prepare_dataset(config: Config):
    train_caption_file = '../../coco2014_train_' + config.preprocess_mode + '.pkl'
    validate_caption_file = '../../coco2014_val_' + config.preprocess_mode + '.pkl'
    test_caption_file = '../../coco2014_test_' + config.preprocess_mode + '.pkl'
    vocabulary_file = '../../vocabulary-' + str(config.word_frequency_threshold) + '_' + config.preprocess_mode + '.pkl'

    if config.preprocess_mode == 'standard':
        if not os.path.exists(train_caption_file) or not os.path.exists(validate_caption_file) or not os.path.exists(test_caption_file):
            _CocoCaptions.generate_standard_caption_file(config.dataset_root, 'coco_train_v2.csv', 'coco_val_v2.csv', 'coco_test_v2.csv', train_caption_file, validate_caption_file, test_caption_file)
    elif config.preprocess_mode == 'default':
        if not os.path.exists(train_caption_file) or not os.path.exists(validate_caption_file):
            _CocoCaptions.generate_default_caption_file(config.dataset_root, os.path.join(config.dataset_root, 'dataset.json'), train_caption_file, validate_caption_file)
    else:
        if not os.path.exists(train_caption_file):
            _CocoCaptions.generate_nltk_caption_file(os.path.join(config.dataset_root, 'train', 'train2014'), os.path.join(config.dataset_root, 'train', 'captions_train2014.json'), train_caption_file)
        if not os.path.exists(validate_caption_file):
            _CocoCaptions.generate_nltk_caption_file(os.path.join(config.dataset_root, 'val', 'val2014'), os.path.join(config.dataset_root, 'val', 'captions_val2014.json'), validate_caption_file)
    if not os.path.exists(vocabulary_file):
        _CocoCaptions.generate_vocabulary(train_caption_file, config.word_frequency_threshold, vocabulary_file)

    return train_caption_file, validate_caption_file, test_caption_file, vocabulary_file

def optimizer_step(optimizer, config, epoch):
    current_lr = config.lr * (config.lr_decay ** (epoch // config.lr_decay_epoch))
    for group in optimizer.param_groups:
        group['lr'] = current_lr
    return current_lr


# ref: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
def Gumbel_softmax_sample(logits, temperature):
    U = torch.rand(logits.size()).cuda()
    y = logits - torch.log(-torch.log(U + 1e-20) + 1e-20)
    return F.log_softmax(y / temperature, dim=-1)

# ref: https://github.com/ruotianluo/self-critical.pytorch
def sample_next_word(logprobs, sample_method='', temperature=1):
    if sample_method == 'greedy':
        sample_logprobs, sample_index = torch.max(logprobs.data, 1)
        sample_index = sample_index.view(-1).long()
    elif sample_method == 'Gumbel':
        _, sample_index = torch.max(Gumbel_softmax_sample(logprobs, temperature).data, 1)
        sample_logprobs = logprobs.gather(1, sample_index.unsqueeze(1))
    else:
        if sample_method.startswith('top'):
            top_num = float(sample_method[3:])
            if 0 < top_num < 1:
                probs = F.softmax(logprobs, dim=1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                _cumsum = sorted_probs.cumsum(1)
                mask = _cumsum < top_num
                mask = torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
                sorted_probs = sorted_probs * mask.float()
                sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                logprobs.scatter_(1, sorted_indices, sorted_probs.log())
            else:
                tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                topk, indices = torch.topk(logprobs, int(top_num), dim=1)
                tmp = tmp.scatter(1, indices, topk)
                logprobs = tmp
        sample_index = Categorical(logits=logprobs.detach()).sample()
        sample_logprobs = logprobs.gather(1, sample_index.unsqueeze(1))
    return sample_index, sample_logprobs
