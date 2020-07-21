import sys
sys.path.append('../../pycocoevalcap/bleu')
sys.path.append('../../pycocoevalcap/rouge')
sys.path.append('../../pycocoevalcap/meteor')
sys.path.append('../../pycocoevalcap/cider')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
import os
import pickle
import time
from _coco import _CocoCaptions
import torch
from torch.utils.data import DataLoader


class Evaluator(object):
    def __init__(self, caption_file, stoi, itos, max_sentence_length, transforms, verbosity=1):
        self.BLEU_evaluator = Bleu(n=4)
        self.Rouge_evaluator = Rouge()
        self.Meteor_evaluator = Meteor()
        self.Cider_evaluator = Cider()

        self.caption_file = caption_file
        self.references = pickle.load(open(self.caption_file, 'rb'))
        for image_path in self.references:
            for i in range(len(self.references[image_path])):
                self.references[image_path][i] = ' '.join(self.references[image_path][i])
        self.hypotheses = {image_path: [''] for image_path in self.references}
        self.stoi = stoi
        self.itos = itos
        self.max_sentence_length = max_sentence_length
        self.transforms = transforms
        self.verbosity = verbosity
        self.data = _CocoCaptions(self.caption_file, 'inference', self.stoi, self.max_sentence_length, transforms=self.transforms)
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)

    def evaluate(self, encoder, generator, model_info=''):
        start_time = time.time()
        if not os.path.exists(generator.model_name + '_evaluate'):
            os.mkdir(generator.model_name + '_evaluate')
        result_file = os.path.join(generator.model_name + '_evaluate', 'evaluate.txt')

        with torch.no_grad():
            encoder.eval()
            generator.eval()
            for (images, image_path_index) in self.dataloader:
                images = images.cuda()
                image_feature, mean_image_feature = encoder(images)
                result = generator.decode(image_feature, mean_image_feature)
                for i in range(len(result)):
                    result[i] = self.itos[result[i]]
                self.hypotheses[self.data.image_path[image_path_index[0]]][0] = ' '.join(result)

        [BLEU_1, BLEU_2, BLEU_3, BLEU_4], _ = self.BLEU_evaluator.compute_score(self.references, self.hypotheses)
        ROUGE, _ = self.Rouge_evaluator.compute_score(self.references, self.hypotheses)
        METEOR, _ = self.Meteor_evaluator.compute_score(self.references, self.hypotheses)
        CIDEr, _ = self.Cider_evaluator.compute_score(self.references, self.hypotheses)

        end_time = time.time()
        if self.verbosity > 0:
            if model_info != '':
                print(model_info)
            print('Evaluate time : %.3fs.' % (end_time - start_time))
            print('BLEU-1 :', BLEU_1)
            print('BLEU-2 :', BLEU_2)
            print('BLEU-3 :', BLEU_3)
            print('BLEU-4 :', BLEU_4)
            print('ROUGE :', ROUGE)
            print('METEOR :', METEOR)
            print('CIDEr :', CIDEr)
        with open(result_file, 'w') as evaluate_log_file:
            evaluate_log_file.write(model_info)
            evaluate_log_file.write('Evaluate time : %.3fs.\n' % (end_time - start_time))
            evaluate_log_file.write('BLEU-1 :' + str(BLEU_1) + '\n')
            evaluate_log_file.write('BLEU-2 :' + str(BLEU_2) + '\n')
            evaluate_log_file.write('BLEU-3 :' + str(BLEU_3) + '\n')
            evaluate_log_file.write('BLEU-4 :' + str(BLEU_4) + '\n')
            evaluate_log_file.write('ROUGE :' + str(ROUGE) + '\n')
            evaluate_log_file.write('METEOR :' + str(METEOR) + '\n')
            evaluate_log_file.write('CIDEr :' + str(CIDEr) + '\n')

        return BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, METEOR, CIDEr

    def decode(self, encoder, generator, model_info=''):
        start_time = time.time()
        if not os.path.exists(generator.model_name + '_decode'):
            os.mkdir(generator.model_name + '_decode')
        result_file = os.path.join(generator.model_name + '_decode', 'decode.txt')

        with open(result_file, 'w') as decode_result_file:
            with torch.no_grad():
                encoder.eval()
                generator.eval()
                for (images, image_path_index) in self.dataloader:
                    images = images.cuda()
                    image_feature, mean_image_feature = encoder(images)
                    result = generator.decode(image_feature, mean_image_feature)
                    for i in range(len(result)):
                        result[i] = self.itos[result[i]]
                    image_path = self.data.image_path[image_path_index[0]]
                    hypothesis_caption = ' '.join(result)
                    decode_result_file.write('%s : %s\n' % (image_path, hypothesis_caption))
                    self.hypotheses[image_path][0] = hypothesis_caption

            [BLEU_1, BLEU_2, BLEU_3, BLEU_4], _ = self.BLEU_evaluator.compute_score(self.references, self.hypotheses)
            ROUGE, _ = self.Rouge_evaluator.compute_score(self.references, self.hypotheses)
            METEOR, _ = self.Meteor_evaluator.compute_score(self.references, self.hypotheses)
            CIDEr, _ = self.Cider_evaluator.compute_score(self.references, self.hypotheses)

            end_time = time.time()
            if self.verbosity > 0:
                if model_info != '':
                    print(model_info)
                print('Decode time : %.3fs.' % (end_time - start_time))
                print('BLEU-1 :', BLEU_1)
                print('BLEU-2 :', BLEU_2)
                print('BLEU-3 :', BLEU_3)
                print('BLEU-4 :', BLEU_4)
                print('ROUGE :', ROUGE)
                print('METEOR :', METEOR)
                print('CIDEr :', CIDEr)
                decode_result_file.write(model_info)
                decode_result_file.write('\n\nDecode time : %.3fs.\n' % (end_time - start_time))
                decode_result_file.write('BLEU-1 :' + str(BLEU_1) + '\n')
                decode_result_file.write('BLEU-2 :' + str(BLEU_2) + '\n')
                decode_result_file.write('BLEU-3 :' + str(BLEU_3) + '\n')
                decode_result_file.write('BLEU-4 :' + str(BLEU_4) + '\n')
                decode_result_file.write('ROUGE :' + str(ROUGE) + '\n')
                decode_result_file.write('METEOR :' + str(METEOR) + '\n')
                decode_result_file.write('CIDEr :' + str(CIDEr) + '\n')

        return BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, METEOR, CIDEr
