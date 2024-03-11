import os
from os.path import join, exists
import h5py
import numpy as np
from sklearn.preprocessing import normalize
from loguru import logger


def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    # print(s.dtype,s, start.dtype)
    # print(s,e)
    inter = np.minimum(end, e) - np.maximum(start, s)
    # inter = end.min(e) - start.max(s)
    # The line `# union = end.max(e) - start.min(s)` is calculating the union of two sets of values
    # represented by `end` and `start`. It is finding the maximum value between `end` and `e`, and
    # then subtracting the minimum value between `start` and `s`. This operation is commonly used in
    # calculating the Intersection over Union (IoU) metric for bounding boxes in object detection
    # tasks.
    union = np.maximum(end, e) - np.minimum(start, s)
    # union = end.max(e) - start.min(s)
    return np.maximum(inter, 0) / union






def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()   
    
    grids = np.stack([grids[0], grids[1]], axis=1)
    # logger.info(f'grids: {grids.shape}')
    scores = score2d[grids[:, 0], grids[:, 1]]
    # logger.info(f'scores: {scores.shape}')
    grids[:, 1] += 1
    # logger.info(f'duration: {duration}, num_clips: {num_clips}')
    moments = grids * duration / num_clips
    return moments, scores


def moment_to_iou2d(moment, num_clips, duration):
    iou2d = np.ones((num_clips, num_clips))
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d


def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.shape[0]
    idxs = np.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = np.minimum(idxs.round(),num_src_clips-1).astype(int)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            meanfeats.append(feats[s:e].mean(axis=0))
        else:
            meanfeats.append(feats[s])
    return np.stack(meanfeats)

def maxfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = np.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    maxfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            maxfeats.append(feats[s:e].max(axis=0)[0])
        else:
            maxfeats.append(feats[s])
    return np.stack(maxfeats)
    
def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid]['c3d_features'][:]
            else:
                feat = f[vid][:]
            feat = normalize(feat)
            vid_feats[vid] = avgfeats(feat, num_pre_clips) 
    return vid_feats
# def normalize(array,axis=1):
    

def get_vid_feat(feat_file, vid, num_pre_clips, dataset_name):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        if dataset_name == "activitynet":
            feat = f[vid]['c3d_features'][:]
            feat = normalize(feat)
        elif dataset_name == "charades":
            feat = f[vid][:]
            feat = normalize(feat)
        elif dataset_name == "didemo":
            feat = f[vid][:200]
            feat = normalize(feat)
        else:
            feat = f[vid][:]
            feat = normalize(feat)

    return avgfeats(feat, num_pre_clips)

def get_feat_didemo(feat_file, vid):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        feat = f[vid][:]
    return Tensor(feat)

def get_c3d_charades(feat_file, num_pre_clips):
    assert exists(feat_file)
    # TODO: fix the loading
    feat = torch.load(feat_file)
    #feat = F.normalize(feat, dim=1)
    return maxfeats(feat, num_pre_clips)

def bert_embedding(sentence, tokenizer):
    query_token = tokenizer(sentence,max_length=128, padding="max_length",return_tensors='ms')
    # print(query_token)
    word_lens = query_token['attention_mask'].sum(axis=1)
    queries = query_token['input_ids']
    return queries, word_lens

def bert_embedding_batch(sentences, tokenizer):
    queries = []
    word_lens = []
    for sentence in sentences:
        query_token = tokenizer(sentence, max_length=128, padding="max_length",return_tensors='ms')
        word_lens.append(query_token['attention_mask'].sum(axis=1))
        queries.append(query_token['input_ids'])
    return queries, word_lens

#TODO: fix the glove embedding
def glove_embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        embedding, vocab = Glove.from_pretrained('80B', 300, special_tokens=["<unk>", "<pad>"])
        # vocab.itos.extend(['<unk>'])
        # vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        # vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        # vocabs.append(vocab)
    
    if len(embedders) == 0:
        embedder = nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)
    
    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = Tensor([vocab.stoi.get(w.lower(), 400000) for w in sentence.split()]).long()
    return embedder(word_idxs)
