from copyreg import pickle
from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import pickle
from mindspore import Tensor, ops
from loguru import logger
import torch
import numpy as np


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()   
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    # print(s.dtype,s, start.dtype)
    # print(s,e)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def nms(moments, scores, thresh):
    cores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed]


def evaluate(cfg, dataset, predictions, nms_thresh, recall_metrics=(1, 5)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.3, 0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
    dataset_name = dataset.__class__.__name__
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    num_clips = predictions[0]['iou'].shape[-1]
    table = [['R@{},IoU@{:.01f}'.format(i, (torch.round(j*100)/100)) for i in recall_metrics for j in iou_metrics]]
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    num_instance = 0
    num_video = 0
    mious_ = []
    lengths = []
    durations = []
    dataset = dataset.create_dict_iterator()
    # for idx, result2d in tqdm(enumerate(predictions)): 
    for idx, data in enumerate(dataset): # each batch  
        sen_l_batch= data['sen_len'].asnumpy()
        duration_batch = data['duration'].asnumpy()
        gt_moments_batch = data['moment'].asnumpy()
        sentences_batch = data['sentence'].asnumpy()
        # logger.info(f'sen_l_batch: {sen_l_batch}, sentences: {len(sentences_batch)}, gt_moments: {gt_moments_batch.shape}, {duration_batch}')
        
        # raise
        for i,l in enumerate(sen_l_batch): # each video
            result2d = predictions[num_video]
            num_video += 1
            # logger.info(f"result2d['contrastive']: {type(result2d['contrastive'])}, {type(result2d['iou'])}")
            score2d_video = torch.pow(torch.FloatTensor(result2d['contrastive']) * 0.5 + 0.5, cfg.TEST.CONTRASTIVE_SCORE_POW) * torch.FloatTensor(result2d['iou'])
            # logger.info(f'score2d_video {score2d_video.shape}')
            gt_moment_video = gt_moments_batch[i][:l]
            sentence_video = sentences_batch[i][:l]
            duration_video = duration_batch[i]
            # logger.info(f'gt_moment_video: {gt_moment_video.shape}, {duration_video}, {len(sentence_video)}')
            for gt_moment, pred_score2d, sentence in zip(gt_moment_video, score2d_video, sentence_video):  # each sentence
                # logger.info(f'pred_score2d: {type(pred_score2d)}, {num_clips}, {duration_video}')
                # logger.info(f'pred_score2d: {pred_score2d.shape}')
                candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration_video)
                moments = nms(candidates, scores, nms_thresh)
                moments = torch.FloatTensor(moments)
                gt_moment = torch.FloatTensor(gt_moment)
                num_instance+=1
                for i, r in enumerate(recall_metrics):
                    # print('gt',gt_moment.shape,moments[:r].shape)
                    mious = iou(moments[:r], gt_moment)
                    # mious = torch.FloatTensor(mious)
                    bools = mious[:, None].expand(r, num_iou_metrics) >= iou_metrics
                    # bools = torch.Tensor(bools)
                    recall_x_iou[i] += bools.any(dim=0)
                miou = iou(moments[:1], gt_moment)
                mious_.append(float(miou))
                lengths.append(float(gt_moment[1] - gt_moment[0]))
                durations.append(float(duration_video))
       
    with open('outputs.pkl', 'wb') as f:
        pickle.dump({
            'mious': mious_, 
            'lengths': lengths,
            'durations': durations,
        }, f)
    recall_x_iou /= num_instance
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('mIoU: %.2f\n'%(sum(mious_) / len(mious_) * 100) + table.table)
    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
    best_r1 = sum(recall_x_iou[0])/num_iou_metrics
    best_r5 = sum(recall_x_iou[1])/num_iou_metrics
    result_dict['Best_R1'] = best_r1
    result_dict['Best_R5'] = best_r5
    return result_dict
