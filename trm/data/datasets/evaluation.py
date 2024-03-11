from copyreg import pickle
from terminaltables import AsciiTable
from tqdm import tqdm
import logging
from trm.data.datasets.utils import iou, score2d_to_moments_scores
import pickle
from mindspore import Tensor, ops
from loguru import logger

def nms(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True)
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
    recall_metrics = Tensor(recall_metrics)
    iou_metrics = Tensor(iou_metrics)
    num_clips = predictions[0]['iou'].shape[-1]
    table = [['R@{},IoU@{:.01f}'.format(i, (ops.round(j*100)/100).asnumpy()) for i in recall_metrics for j in iou_metrics]]
    recall_x_iou = ops.zeros((num_recall_metrics, num_iou_metrics))
    num_instance = 0
    mious_ = []
    lengths = []
    durations = []
    dataset = dataset.create_dict_iterator()
    for idx, data in enumerate(dataset):   # each video
        result2d = predictions[idx]
        sen_l = data['sen_len']
        score2d = result2d['iou']
        duration = data['duration']
        gt_moments = data['moment']
        sentences = data['sentence']
        logger.info(f'sen_l: {sen_l}, sentences: {len(sentences)}, gt_moments: {len(gt_moments)}, iou shape: {score2d.shape}')
        # logger.info(f"result2d['contrastive']: {result2d['contrastive']}")
        # raise
        scorec2d = ops.pow(Tensor(result2d['contrastive'] * 0.5 + 0.5), cfg.TEST.CONTRASTIVE_SCORE_POW) * result2d['iou']
        for gt_moment, pred_score2d, sentence,l in zip(gt_moments, score2d, sentences,sen_l):  # each sentence
            num_instance += 1
            logger.info(f'pred_score2d:{pred_score2d.shape}')
            logger.info(f'num_clips: {num_clips}, duration: {duration}')
            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
            moments = nms(candidates, scores, nms_thresh)
            for i, r in enumerate(recall_metrics):
                mious = iou(moments[:r], gt_moment)
                bools = mious[:, None].expand(r, num_iou_metrics) >= iou_metrics
                recall_x_iou[i] += bools.any(dim=0)
            miou = iou(moments[:1], gt_moment)
            mious_.append(float(miou))
            lengths.append(float(gt_moment[1] - gt_moment[0]))
            durations.append(float(duration))
    # with open('outputs.pkl', 'wb') as f:
    #     pickle.dump({
    #         'mious': mious_, 
    #         'lengths': lengths,
    #         'durations': durations,
    #     }, f)
    recall_x_iou /= num_instance
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('mIoU: %.2f\n'%(sum(mious_) / len(mious_) * 100) + table.table)
    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], ops.round(iou_metrics[j]*100)/100).asnumpy()] = recall_x_iou[i][j]
    best_r1 = sum(recall_x_iou[0])/num_iou_metrics
    best_r5 = sum(recall_x_iou[1])/num_iou_metrics
    result_dict['Best_R1'] = best_r1
    result_dict['Best_R5'] = best_r5
    return result_dict

