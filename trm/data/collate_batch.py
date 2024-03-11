from trm.structures import TLGBatch
import mindspore.numpy as np

class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats, queries, wordlens, ious2d, moments, num_sentence, idxs, sentences, durations, phrase = transposed_batch

        return TLGBatch(
            feats=np.stack(feats).float(),
            queries=queries,
            wordlens=wordlens,
            all_iou2d=ious2d,
            moments=moments,
            num_sentence=num_sentence,
            sentences=sentences,
            durations=durations,
            phrase=phrase,
        ), idxs
