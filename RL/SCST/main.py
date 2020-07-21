import sys
sys.path.append('../../')
import os
import time
import random
import re
from tqdm import tqdm
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from util import Config
from util import prepare_dataset
from util import Evaluator
from _coco import _CocoCaptions
from models import ShowTell
from models import ShowAttendTell
from models import AdaptiveAttention


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


config = Config()
train_caption_file, validate_caption_file, test_caption_file, vocabulary_file = prepare_dataset(config)
max_sentence_length = _CocoCaptions.get_max_sentence_length(train_caption_file)
coco_itos, coco_stoi, coco_vectors = _CocoCaptions.get_coco_dict_vectors(10000000, config.word_embedding_dim, vocabulary_file)
vocabulary_size = coco_vectors.size(0)
config.vocabulary_size = vocabulary_size
print('Vocabulary size : %d' % vocabulary_size)
config.max_sentence_length = max_sentence_length
transforms = config.transforms


def train(model):
    model.cuda()
    model.initialize(coco_vectors)
    generator_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam([{'params': generator_parameters}], lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    train_data = _CocoCaptions(train_caption_file, 'train', coco_stoi, max_sentence_length, transforms=transforms)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size // 8)
    if not os.path.exists(model.model_name + '_log'):
        os.mkdir(model.model_name + '_log')
    if not os.path.exists(model.model_name + '_models'):
        os.mkdir(model.model_name + '_models')
    if not os.path.exists(model.model_name + '_model'):
        os.mkdir(model.model_name + '_model')
    writer = SummaryWriter(log_dir=model.model_name + '_log', comment=model.model_name + '_train', filename_suffix='.train')
    iteration = 0
    iteration_sample_num = 0
    iteration_loss = 0

    evaluator = Evaluator(validate_caption_file, coco_stoi, coco_itos, max_sentence_length, transforms)
    best_CIDEr = 0
    no_improve = 0
    best_epoch = 0
    print('Training model :', model.model_name)

    for e in tqdm(range(config.epoch)):
        epoch_sample_num = 0
        epoch_loss = 0
        model.train()
        for (images, target, target_mask) in train_dataloader:
            images = images.cuda()                                                                                                  # [batch_size, channel, height, width]
            target = target.cuda()                                                                                                  # [batch_size, max_sentence_length]
            target_mask = target_mask.cuda()                                                                                        # [batch_size, max_sentence_length]
            target_len = target_mask.sum(dim=1, keepdim=False)                                                                      # [batch_size]
            sample_num = images.size(0)

            log_probs, attention_weights = model(images, target, max_step=(int)(target_len.max()))                                  # [batch_size, max_sentence_length, vocabulary_size]
            generation_loss = (torch.gather(-log_probs, 2, target.unsqueeze(2)).squeeze(2) * target_mask).sum(dim=1, keepdim=False) # [batch_size]
            loss = (generation_loss / target_len).mean() if config.average_sentence_loss else generation_loss.mean()
            if config.coverage:
                coverage_loss = torch.pow(1 - attention_weights.sum(dim=1, keepdim=False), 2).mean()
                loss += config.Lambda * coverage_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            iteration_sample_num += sample_num
            iteration_loss += loss
            epoch_sample_num += sample_num
            epoch_loss += loss
            iteration += 1
            if iteration % config.iteration_to_show == 0:
                writer.add_scalar('Train loss (iteration)', float(iteration_loss) / iteration_sample_num, iteration)
                iteration_loss = 0
                iteration_sample_num = 0

        print('Training epoch %d : loss = %.3f' % (e + 1, float(epoch_loss) / epoch_sample_num))
        writer.add_scalar('Train loss (epoch)', float(epoch_loss) / epoch_sample_num, e + 1)
        BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, METEOR, CIDEr = evaluator.evaluate(model, model_info='validate_epoch-' + str(e + 1))
        writer.add_scalar('BLEU-1', BLEU_1, e + 1)
        writer.add_scalar('BLEU-2', BLEU_2, e + 1)
        writer.add_scalar('BLEU-3', BLEU_3, e + 1)
        writer.add_scalar('BLEU-4', BLEU_4, e + 1)
        writer.add_scalar('ROUGE', ROUGE, e + 1)
        writer.add_scalar('METEOR', METEOR, e + 1)
        writer.add_scalar('CIDEr', CIDEr, e + 1)
        torch.save({
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float(epoch_loss) / epoch_sample_num,
            'BLEU-1': BLEU_1,
            'BLEU-2': BLEU_2,
            'BLEU-3': BLEU_3,
            'BLEU-4': BLEU_4,
            'ROUGE': ROUGE,
            'METEOR': METEOR,
            'CIDEr': CIDEr
        }, model.model_name + '_models/' + model.model_name + '-' + str(e + 1))
        if CIDEr > best_CIDEr:
            best_CIDEr = CIDEr
            no_improve = 0
            best_epoch = e + 1
        else:
            no_improve += 1
            if no_improve >= config.early_stopping_epoch:
                break

        scheduler.step()

    print('Best epoch : %d\nBest validation CIDEr : %.3f' % (best_epoch, best_CIDEr))
    shutil.copy(model.model_name + '_models/' + model.model_name + '-' + str(best_epoch), model.model_name + '_model/' + model.model_name)


def load_model(model_path=''):
    model = ShowAttendTell(config)
    if model_path == '':
        assert os.path.exists(model.model_name + '_models'), 'when default models not exist, model path can not be empty'
        max_model_index = -1
        for model_file in os.listdir(model.model_name + '_models'):
            if os.path.isfile(os.path.join(model.model_name + '_models', model_file)) and model_file[:len(model.model_name) + 1] == model.model_name + '-':
                model_index = model_file.strip()[len(model.model_name) + 1:]
                if re.match(re.compile(r'\d+'), model_index):
                    max_model_index = int(model_index)
        assert max_model_index != -1, 'models not exist'
        path = os.path.join(model.model_name + '_models', model.model_name + '-' + str(max_model_index))
    else:
        if os.path.exists(os.path.join(model.model_name + '_models', model_path)):
            path = os.path.join(model.model_name + '_models', model_path)
        elif os.path.exists(model_path):
            path = model_path
        else:
            raise Exception('model not found at %s' % model_path)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])
    model.cuda()
    return model

def test(model_path):
    model = load_model(model_path)
    evaluator = Evaluator(test_caption_file, coco_stoi, coco_itos, max_sentence_length, transforms)
    print('Testing model :', model.model_name)
    BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, METEOR, CIDEr = evaluator.evaluate(model, model_info='test-' + model_path)

def decode(model_path):
    model = load_model(model_path)
    evaluator = Evaluator(test_caption_file, coco_stoi, coco_itos, max_sentence_length, transforms)
    print('Decoding model :', model.model_name)
    BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, METEOR, CIDEr = evaluator.decode(model, model_info='test-' + model_path)

def watch(model):
    assert os.path.exists(model.model_name + '_models'), 'model path ' + model.model_name + '_models not exists'
    model.cuda()
    evaluator = Evaluator(test_caption_file, coco_stoi, coco_itos, max_sentence_length, transforms)
    sleep_time = 60
    watch_index = 0
    writer = SummaryWriter(log_dir=model.model_name + '_log', comment=model.model_name + '_watch', filename_suffix='.watch')
    print('Testing model :', model.model_name)
    while True:
        path = ''
        for model_file in os.listdir(model.model_name + '_models'):
            if os.path.isfile(os.path.join(model.model_name + '_models', model_file)) and model_file[:len(model.model_name) + 1] == model.model_name + '-':
                model_index = int(model_file.strip()[len(model.model_name) + 1:])
                if model_index == watch_index + config.epoch_to_save:
                    watch_index = model_index
                    path = model_file
                    break
        if path != '':
            print('Validating ' + path)
            model.load_state_dict(torch.load(os.path.join(model.model_name + '_models', path), map_location=torch.device('cpu'))['model_state_dict'])
            BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, METEOR, CIDEr = evaluator.evaluate(model, model_info='validate_epoch-' + str(model_index))
            if watch_index == config.epoch // config.epoch_to_save * config.epoch_to_save:
                print('Watch ended at model-' + str(watch_index))
                break
        else:
            time.sleep(sleep_time)


if __name__ == '__main__':
    if config.mode == 'train':
        model = ShowAttendTell(config)
        train(model)
        test(model.model_name + '_model/' + model.model_name)
    elif config.mode == 'test':
        test('show_attend_tell_model/show_attend_tell')
    elif config.mode == 'decode':
        decode('show_attend_tell_model/show_attend_tell')
    elif config.mode == 'watch':
        model = ShowAttendTell(config)
        watch(model)
