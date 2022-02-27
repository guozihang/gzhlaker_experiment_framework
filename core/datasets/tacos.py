import json
import logging
import torch
from torch.utils.data import Dataset
from .utils import moment_to_iou2d, bert_embedding, get_vid_feat, clip_embedding
from transformers import DistilBertTokenizer
from clip import clip


class TACoSDataset(Dataset):

    def __init__(self, ann_file, feat_file, num_pre_clips, num_clips):
        super(TACoSDataset, self).__init__()
        self.annos = []
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        self._get_annotation()


        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        for vid, anno in annos.items():
            duration = anno['num_frames'] / anno['fps']  # duration of the video
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor(
                        [max(timestamp[0] / anno['fps'], 0), min(timestamp[1] / anno['fps'], duration)])
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)

            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)
            # queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
            queries, word_lens = clip_embedding(sentences)

            # assert moments.size(0) == all_iou2d.size(0)
            # assert moments.size(0) == queries.size(0)
            # # assert moments.size(0) == word_lens.size(0)
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,  # N * 2
                    'iou2d': all_iou2d,  # N * 128*128
                    'sentence': sentences,  # list, len=N
                    'query': queries,  # padded query, N*word_len*C for LSTM and N*word_len for BERT
                    'wordlen': word_lens,  # size = N
                    'duration': duration
                }
            )

        # self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="tacos")

    def __getitem__(self, idx):
        # feat = self.feats[self.annos[idx]['vid']]
        feat_c3d, feat_clip = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips,
                                           dataset_name="tacos")
        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        iou2d = self.annos[idx]['iou2d']
        moment = self.annos[idx]['moment']
        return feat_c3d, feat_clip, query, wordlen, iou2d, moment, len(self.annos[idx]['sentence']), idx

    def __len__(self):
        return len(self.annos)

    def _get_annotation(self):
        with open(self.ann_file, 'r') as f:
            self.annotations = json.load(file)

    def _get_video_feature(self):
        pass

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']
