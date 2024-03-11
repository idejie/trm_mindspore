from dataclasses import dataclass
import mindspore

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: mindspore.Tensor 
    queries: list
    wordlens: list
    all_iou2d: list
    moments: list
    num_sentence: list
    sentences: list
    durations: list
    phrase: list

    # def to(self, device):
    #     # self.frames = [f.to(device) for f in self.frames]
    #     self.feats = self.feats.to(device)
    #     self.queries = [query.to(device) for query in self.queries]
    #     self.wordlens = [word_len.to(device) for word_len in self.wordlens]
    #     self.all_iou2d = [iou2d.to(device) for iou2d in self.all_iou2d]
    #     self.moments = [moment.to(device) for moment in self.moments]

    #     return self
    

