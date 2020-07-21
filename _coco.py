from torchtext.vocab import GloVe
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy as np
import random
import torch
import os
import re
from nltk.tokenize import word_tokenize
import pickle
import json
import csv


class _CocoCaptions(VisionDataset):
    # Captions splited and formalized as [train: 113287, val: 5000, test: 5000]
    @staticmethod
    def generate_standard_caption_file(dataset_root, train_csv_file, val_csv_file, test_csv_file, train_caption_file, val_caption_file, test_caption_file):
        train_caption_list = dict()
        val_caption_list = dict()
        test_caption_list = dict()
        with open(os.path.join(dataset_root, train_csv_file)) as train_csv, open(os.path.join(dataset_root, val_csv_file)) as val_csv, open(os.path.join(dataset_root, test_csv_file)) as test_csv:
            train_reader = csv.DictReader(train_csv)
            val_reader = csv.DictReader(val_csv)
            test_reader = csv.DictReader(test_csv)
            for row in train_reader:
                captions = row['Captions'].split('---')
                if row['FileName'][5:10] == 'train':
                    train_caption_list[os.path.join(dataset_root, 'train', 'train2014', row['FileName'])] = [word_tokenize(captions[i]) for i in range(5)]
                else:
                    train_caption_list[os.path.join(dataset_root, 'val', 'val2014', row['FileName'])] = [word_tokenize(captions[i]) for i in range(5)]
            for row in val_reader:
                captions = row['Captions'].split('---')
                assert row['FileName'][5:8] == 'val'
                val_caption_list[os.path.join(dataset_root, 'val', 'val2014', row['FileName'])] = [word_tokenize(captions[i]) for i in range(5)]
            for row in test_reader:
                captions = row['Captions'].split('---')
                assert row['FileName'][5:8] == 'val'
                test_caption_list[os.path.join(dataset_root, 'val', 'val2014', row['FileName'])] = [word_tokenize(captions[i]) for i in range(5)]
        with open(train_caption_file, 'wb') as train_f, open(val_caption_file, 'wb') as val_f, open(test_caption_file, 'wb') as test_f:
            pickle.dump(train_caption_list, train_f)
            pickle.dump(val_caption_list, val_f)
            pickle.dump(test_caption_list, test_f)

    # Captions splited and formalized as [https://cs.stanford.edu/people/karpathy/deepimagesent/]
    @staticmethod
    def generate_default_caption_file(dataset_root, json_file, train_caption_file, val_caption_file):
        train_caption_list = dict()
        val_caption_list = dict()
        with open(json_file) as f:
            data_list = json.loads(f.read())['images']
            for data in data_list:
                captions = [data['sentences'][i]['tokens'] + ['.'] for i in range(5)]
                if data['filepath'] == 'train2014':
                    train_caption_list[os.path.join(dataset_root, 'train', 'train2014', data['filename'])] = captions
                else:
                    val_caption_list[os.path.join(dataset_root, 'val', 'val2014', data['filename'])] = captions
        with open(train_caption_file, 'wb') as train_f, open(val_caption_file, 'wb') as val_f:
            pickle.dump(train_caption_list, train_f)
            pickle.dump(val_caption_list, val_f)

    # Captions splited and formalized by NLTK
    @staticmethod
    def generate_nltk_caption_file(dataset_root, annFile, caption_file):
        from pycocotools.coco import COCO
        coco = COCO(annFile)
        ids = list(sorted(coco.imgs.keys()))
        caption_list = dict()
        bad_format_sentences = {
            # train dataset
            'A worn sign that reads \"STOP: NO THRU TRAFFIC)' : 'A worn sign that reads \"STOP: NO THRU TRAFFIC\"',
            'A view of a keyboard and mouse, on a table that reads \"workstation:.' : 'A view of a keyboard and mouse, on a table that reads \"workstation\".',
            'A red stop sign sitting on the side of a road with the word \" Driving \' written under it.' : 'A red stop sign sitting on the side of a road with the word \" Driving \" written under it.',
            'Several surfboards roped off with \'Surf Art Expert\" sign' : 'Several surfboards roped off with \"Surf Art Expert\" sign',
            'A stop sign with a \"driving\' sticker placed on it.' : 'A stop sign with a \"driving\" sticker placed on it.',
            'A man wearing a tie written \'Why knot?\" with a checkered blazer.' : 'A man wearing a tie written \"Why knot?\" with a checkered blazer.',
            'Dense cabbage plants with some of their \"flowers\'.' : 'Dense cabbage plants with some of their \"flowers\".',
            'A bench dedicated to the memory of Lloyd \'n\" Milly.' : 'A bench dedicated to the memory of Lloyd \"n\" Milly.',
            'A \"\"LAN\" Brand airplane at an airport near the sea.' : 'A \"LAN\" Brand airplane at an airport near the sea.',
            # val dataset
            'A man is holding a frisbee that reads \"queer\' on it.' : 'A man is holding a frisbee that reads \"queer\" on it.',
            'A small dog on TV behind the words \" What Did I Do Wrong?\'.' : 'A small dog on TV behind the words \" What Did I Do Wrong?\".',
            'Box for 6\" Bench Grinder sitting next to a knife.' : 'Box for \"6 Bench Grinder\" sitting next to a knife.',
            'Teddy bears are on display beneath a ?travel necessities\" sign' : 'Teddy bears are on display beneath a \"travel necessities\" sign',
            'A rural intersection with a stop sign that reads, \'don\'t STOP believing\".' : 'A rural intersection with a stop sign that reads, \"don\'t STOP believing\".'
        }
        quote_pattern1 = re.compile(r'\s+\'(.+?)\'\s+')
        quote_pattern2 = re.compile(r'\s+\"(.+?)\"\s+')
        for img_id in ids:
            image_path = os.path.join(dataset_root, coco.loadImgs(img_id)[0]['file_name'])
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = []
            for i, ann in enumerate(anns):
                if i == 5:
                    break
                caption = ann['caption']
                if caption in bad_format_sentences:
                    caption = bad_format_sentences[caption]
                if caption[-1] == '.':
                    caption = re.sub(quote_pattern1, ' quotesequencemark ', caption[:-1] + ' .')
                else:
                    caption = re.sub(quote_pattern1, ' quotesequencemark ', caption + ' ')
                caption = re.sub(quote_pattern2, ' quotesequencemark ', caption).strip()
                sentence = ''
                quote_flag = False
                for j in range(len(caption)):
                    if caption[j] != '\"':
                        if not quote_flag:
                            sentence += caption[j]
                    else:
                        if quote_flag:
                            sentence += ' quotesequencemark '
                            quote_flag = False
                        else:
                            quote_flag = True
                assert not quote_flag, 'image(%s) : sentence quote format error : %s' % (image_path, caption)
                words = word_tokenize(sentence.lower())
                if words[-1] != '.':
                    if words[-1] in [',', '!', '?', ';', '`']:
                        words[-1] = '.'
                    else:
                        words.append('.')
                captions.append(words)
            caption_list[image_path] = captions
        with open(caption_file, 'wb') as f:
            pickle.dump(caption_list, f)

    @staticmethod
    def generate_vocabulary(caption_file, word_frequency_threshold, vocabulary_file):
        caption_list = pickle.load(open(caption_file, 'rb'))
        vocabulary_cnt = dict()
        for captions in caption_list.values():
            for caption in captions:
                for word in caption:
                    if word not in vocabulary_cnt:
                        vocabulary_cnt[word] = 1
                    else:
                        vocabulary_cnt[word] += 1
        sorted_vocabulary_cnt = sorted(vocabulary_cnt.items(), key=lambda x: x[1], reverse=True)
        filtered_sorted_vocabulary_cnt = list(filter(lambda x: x[1] >= word_frequency_threshold, sorted_vocabulary_cnt))
        vocabulary = dict()
        vocabulary['<eos>'] = 0
        for i, word in enumerate(filtered_sorted_vocabulary_cnt):
            vocabulary[word[0]] = i + 1
        assert '<unk>' not in vocabulary, 'special mark <unk> is already in vocabulary'
        vocabulary['<unk>'] = len(vocabulary)
        with open(vocabulary_file, 'wb') as f:
            pickle.dump({'vocabulary_cnt' : sorted_vocabulary_cnt, 'vocabulary' : vocabulary}, f)

    @staticmethod
    def get_max_sentence_length(caption_file):
        caption_list = pickle.load(open(caption_file, 'rb'))
        max_sentence_length = 0
        for captions in caption_list.values():
            for caption in captions:
                max_sentence_length = max(max_sentence_length, len(caption))
        return max_sentence_length + 1

    @staticmethod
    def get_coco_dict_vectors(word_embedding_num, word_embedding_dim, vocabulary_file):
        glove = GloVe(name='840B', dim=word_embedding_dim, cache='../../glove', max_vectors=word_embedding_num)
        glove_vectors = glove.vectors
        glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
        glove_stoi = glove.stoi
        coco_stoi = pickle.load(open(vocabulary_file, 'rb'))['vocabulary']
        coco_itos = ['' for _ in range(len(coco_stoi))]
        for word in coco_stoi:
            coco_itos[coco_stoi[word]] = word
        coco_vectors = torch.zeros([len(coco_stoi) - 2, word_embedding_dim])
        for i in range(len(coco_stoi) - 2):
            if coco_itos[i] in glove_stoi:
                coco_vectors[i, :] = glove_vectors[glove_stoi[coco_itos[i]]]
            else:
                random_vector = torch.zeros(word_embedding_dim)
                random_vector.normal_(mean=0, std=0.1)
                coco_vectors[i, :] = random_vector + glove_mean_vector
        coco_vectors = torch.cat([coco_vectors, coco_vectors.mean(dim=0, keepdim=True)], dim=0)
        coco_vectors = torch.cat([coco_vectors, torch.zeros([1, word_embedding_dim])], dim=0)
        return coco_itos, coco_stoi, coco_vectors

    def __init__(self, caption_file, mode, stoi, max_sentence_length, transform=None, target_transform=None, transforms=None, flag='default', K=5):
        super(_CocoCaptions, self).__init__(None, transforms, transform, target_transform)
        captions = pickle.load(open(caption_file, 'rb'))
        self.mode = mode.lower()
        self.flag = flag.lower()
        self.K = K
        self.image_path = []
        if self.mode not in ['train', 'inference']:
            raise Exception('mode must in [train, inference]')
        if self.flag not in ['default', 'cl']:
            raise Exception('mode must in [default, CL]')
        image_num = len(captions)

        if self.mode == 'train':
            if self.flag == 'default':
                self.num = image_num * 5
                self.target = np.zeros(shape=[self.num, max_sentence_length], dtype=np.int64)
                self.target_mask = np.zeros(shape=[self.num, max_sentence_length], dtype=np.float32)
                for index, image_path in enumerate(captions):
                    self.image_path.append(image_path)
                    for i in range(5):
                        target_index = index * 5 + i
                        words = captions[image_path][i]
                        word_num = len(words)
                        for j in range(word_num):
                            word = words[j]
                            if word in stoi:
                                self.target[target_index][j] = stoi[word]
                            else:
                                self.target[target_index][j] = stoi['<unk>']
                            self.target_mask[target_index][j] = 1
                        self.target[target_index][word_num] = 0
                        self.target_mask[target_index][word_num] = 1
                print('%s: %d items loaded' % (caption_file, len(captions)))
            else:
                self.num = image_num * 5 * (K + 1)
                self.target = np.zeros(shape=[self.num, max_sentence_length], dtype=np.int64)
                self.target_mask = np.zeros(shape=[self.num, max_sentence_length], dtype=np.float32)
                self.label = np.zeros(shape=[self.num, 1], dtype=np.float32)
                caption_list = [image_path for image_path in captions]
                for index, image_path in enumerate(captions):
                    self.image_path.append(image_path)
                    for i in range(5):
                        target_index = (index * 5 + i) * (K + 1)
                        words = captions[image_path][i]
                        word_num = len(words)
                        for j in range(word_num):
                            word = words[j]
                            if word in stoi:
                                self.target[target_index][j] = stoi[word]
                            else:
                                self.target[target_index][j] = stoi['<unk>']
                            self.target_mask[target_index][j] = 1
                        self.target[target_index][word_num] = 0
                        self.target_mask[target_index][word_num] = 1
                        self.label[target_index][0] = -1
                        for k in range(K):
                            target_index = (index * 5 + i) * (K + 1) + k + 1
                            _index = random.randint(0, image_num - 1)
                            while _index == index:
                                _index = random.randint(0, image_num - 1)
                            __index = random.randint(0, 4)
                            words = captions[caption_list[_index]][__index]
                            word_num = len(words)
                            for j in range(word_num):
                                word = words[j]
                                if word in stoi:
                                    self.target[target_index][j] = stoi[word]
                                else:
                                    self.target[target_index][j] = stoi['<unk>']
                                self.target_mask[target_index][j] = 1
                            self.target[target_index][word_num] = 0
                            self.target_mask[target_index][word_num] = 1
                            self.label[target_index][0] = 1
                print('%s: %d items loaded' % (caption_file, len(captions)))
        else:
            self.num = image_num
            for image_path in captions:
                self.image_path.append(image_path)
            print('%s: %d items loaded' % (caption_file, self.num))

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.flag == 'default':
                img = Image.open(self.image_path[index // 5]).convert('RGB')
                if self.transforms is not None:
                    img = self.transforms(img)
                return img, torch.from_numpy(self.target[index]), torch.from_numpy(self.target_mask[index])
            else:
                img = Image.open(self.image_path[index // (5 * (self.K + 1))]).convert('RGB')
                if self.transforms is not None:
                    img = self.transforms(img)
                return img, torch.from_numpy(self.target[index]), torch.from_numpy(self.target_mask[index]), torch.from_numpy(self.label[index])
        else:
            img = Image.open(self.image_path[index]).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return img, index

    def __len__(self):
        return self.num
