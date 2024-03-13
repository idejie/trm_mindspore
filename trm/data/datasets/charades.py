import os
import json
from loguru import logger
from .utils import moment_to_iou2d,  bert_embedding, get_vid_feat
from mindformers import AutoTokenizer, AutoModel
from mindspore import Tensor, ops
import numpy as np
class CharadesDataset:
    def __init__(self, ann_file, feat_file, num_pre_clips, num_clips, remove_person=False,debug=False):
        # super(CharadesDataset, self).__init__()
        self.ann_name = os.path.basename(ann_file)
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file, 'r') as f:
            annos = json.load(f)
        self.annos = []
        tokenizer = AutoTokenizer.from_pretrained('bert_base_uncased')
        logger.info("Preparing data, please wait...")
        self.remove_person = remove_person

        for vid, anno in annos.items():
            duration = anno['duration']
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            phrases = []
            if 'phrases' not in anno:
                anno['phrases'] = anno['sentences']
            for timestamp, sentence, phrase in zip(anno['timestamps'], anno['sentences'], anno['phrases']):
                if timestamp[0] < timestamp[1]:
                    moment = np.array([max(timestamp[0], 0), min(timestamp[1], duration)],np.float32)
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)
                    new_phrase = []
                    for i in range(len(phrase)):
                        # if self.remove_person:
                        #     if 'person' not in phrase[i] and 'people' not in phrase[i]:
                        #         new_phrase.append(phrase[i])
                        # else:
                        new_phrase.append(phrase[i])
                        # phrase[i] = 'A photo of ' + phrase[i]
                    if len(new_phrase) == 0:
                        new_phrase.append(sentence)
                    # phrase.insert(0, sentence)
                    phrases.append(new_phrase)
            moments = np.stack(moments)
            all_iou2d = np.stack(all_iou2d)
            queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
            queries = queries.asnumpy()
            word_lens = word_lens.asnumpy()
            assert moments.shape[0] == all_iou2d.shape[0]
            assert moments.shape[0] == queries.shape[0]
            assert moments.shape[0] == word_lens.shape[0]

            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'iou2d': all_iou2d,
                    'sentence': sentences,
                    'query': queries,
                    'wordlen': word_lens,
                    'duration': duration,
                    'phrase': phrases,
                }
            )
        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="charades")

    def __getitem__(self, idx):
        #feat = self.feats[self.annos[idx]['vid']]
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="charades")
        return {
            "feature": feat,
            "query": self.annos[idx]['query'],
            "wordlen": self.annos[idx]['wordlen'],
            "iou2d": self.annos[idx]['iou2d'],
            "moment": self.annos[idx]['moment'],
            "num_sentence": len(self.annos[idx]['sentence']),
            "idx": idx,
            "sentence": self.annos[idx]['sentence'],
            "duration": self.annos[idx]['duration'],
            "phrase": self.annos[idx]['phrase'],
        }
        # return feat, self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['moment'], len(self.annos[idx]['sentence']), idx, self.annos[idx]['sentence'], self.annos[idx]['duration'], self.annos[idx]['phrase']

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']






