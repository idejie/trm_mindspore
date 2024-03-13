from trm.utils.imports import import_file
from trm.modeling.trm.text_encoder import pad_sequence
from trm.data.datasets.utils import iou
from . import datasets as D
from .collate_batch import BatchCollator
from mindspore.dataset import GeneratorDataset
from trm.structures import TLGBatch
import numpy as np
from mindspore import ops, Tensor
from loguru import logger
import torch
def build_dataset(dataset_list, dataset_catalog, cfg, is_train=True):
    # build specific dataset
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(
                dataset_list
            )
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["num_pre_clips"] = cfg.INPUT.NUM_PRE_CLIPS
        args["num_clips"] = cfg.MODEL.TRM.NUM_CLIPS
        args["remove_person"] = cfg.MODEL.TRM.LOSS.CONTRASTIVE
        # args['debug'] = cfg.DEBUG
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        raise NotImplementedError('Concatenating multiple datasets is not supported')
    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_train_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
        # TODO: check if drop_last=True helps
    )
    return batch_sampler

def make_test_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )
    return batch_sampler
def  pad_str_sequence(sequences, batch_first=True, padding_value="#"):
    # N x L x sentence
    # N x L x l x phrase
    L = [len(s) for s in sequences]
    if isinstance(sequences[0][0],list):
        max_L = max(L)
        word_num = []
        phrase_num = []
        for data in sequences:
            for sentence in data:
                phrase_num.append(len(sentence))
                for phrase in sentence:
                    word_num.append(len(phrase))
        output = np.full((len(sequences), max_L, max(phrase_num)), padding_value*max(word_num))
        for i, s in enumerate(sequences):
            for j, ss in enumerate(s):
                for k, sss in enumerate(ss):
                    output[i, j,k] = sss
    else:
        max_L = max(L)
        sen_l = [ max([len(ss) for ss in s]) for s in sequences]
        output = np.full((len(sequences), max_L), padding_value*max(sen_l))
        for i, s in enumerate(sequences):
            for j, ss in enumerate(s):
                output[i, j] = ss
    return output
        

def pad_sequence(sequences, batch_first=True, padding_value=0):
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])

    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = np.ones(out_dims) *padding_value 
    
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor
    


    return out_tensor

def batch_process(data,BatchInfo):
    # 1. feature
    feature = [d['feature'] for d in data]
    # logger.info(f'{[f.shape for f in feature]}')
    feature = np.stack(feature)    
    # logger.info(f'feature shape: {feature.shape}')
    # 2.query
    query = [np.array(d['query']) for d in data]
    query = pad_sequence(query)

   
    # logger.info(f'query after: {[q.shape for q in query]}')
    # logger.info(f'{[q.shape for q in query]}')
    # logger.info(f'query shape: {query.shape}')
    # 3.word len
    word_len = [d['wordlen'] for d in data]
    # logger.info(f'word_len: {word_len}')
    word_len = pad_sequence(word_len)
    # 4. iou2d
    iou2d = [d['iou2d'] for d in data]
    # logger.info(f'iou2d: {[i.shape for i in iou2d]}')
    iou2d = pad_sequence(iou2d)
    # logger.info(f'iou2d: {iou2d.shape}')
    # 5. moment
    moment = [d['moment'] for d in data]
    # logger.info(f"moment: {[i.shape for i in moment]}")
    moment = pad_sequence(moment)
    # logger.info(f'moment: {moment.shape}')
    # 6. index
    index = np.array([d['idx'] for d in data])
    # logger.info(f'index: {index}')
    # 7.sentence_len
    sen_len = np.array([d['num_sentence'] for d in data])
    # logger.info(f'sen_len: {sen_len}')
    # 8. duration
    duration = np.array([d['duration'] for d in data])
    # logger.info(f'duration: {duration}')
    # 9. phrase
    phrase = [d['phrase'] for d in data]
    # for i,p in enumerate(phrase):
    #     logger.info(f'phrase-{i}: {p}')
    phrase = pad_str_sequence(phrase)
    # logger.info(f'phrase: {phrase.shape}')
    # logger.info(f'phrase: {[len(p) for p in phrase]}')
    # 10. sentence
    # sentence = np.chararray((len(data),))
    # print(sentence)

    sentence= [d['sentence'] for d in data]
    # for i,p in enumerate(sentence):
    #     logger.info(f'sentence-{i}: {p}')
    sentence = pad_str_sequence(sentence)
    # logger.info(f'sentence: {[len(s) for s in sentence]}')
    # logger.info(f'sentence ({type(sentence[0][0])}): {sentence}')
    return feature,query,word_len,iou2d,moment,index,sen_len, duration,sentence,phrase
    
def make_data_loader(cfg, is_train=True, is_distributed=False, is_for_period=False):
    # num_gpus = get_world_size()
    # if is_train:
    #     batch_size = cfg.SOLVER.BATCH_SIZE
    #     assert (
    #         batch_size % num_gpus == 0
    #     ), "SOLVER.BATCH_SIZE ({}) must be divisible by the number of GPUs ({}) used.".format(
    #         batch_size, num_gpus)
    #     batch_size_per_gpu = batch_size // num_gpus
    #     shuffle = True
    #     max_epoch = cfg.SOLVER.MAX_EPOCH
    # else:
    #     batch_size = cfg.TEST.BATCH_SIZE
    #     assert (
    #         batch_size % num_gpus == 0
    #     ), "TEST.BATCH_SIZE ({}) must be divisible by the number of GPUs ({}) used.".format(
    #         batch_size, num_gpus)
    #     batch_size_per_gpu = batch_size // num_gpus
    #     # shuffle = True if not is_distributed else False  # originally False
    #     shuffle = False

    # if batch_size_per_gpu > 1:
    #     logger = logging.getLogger(__name__)
    logger.info(f'batch_size: {cfg.SOLVER.BATCH_SIZE}')
    paths_catalog = import_file(
        "trm.cfg.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, DatasetCatalog, cfg, is_train=is_train or is_for_period)

    data_loaders = []
    for dataset in datasets:
        dataset = GeneratorDataset(dataset,
                                column_names=['data'],
                                num_parallel_workers=1,
                                shuffle=is_train)
        # dataset = dataset.batch(batch_size=cfg.SOLVER.BATCH_SIZE, input_columns=["'feature','query','word_len','iou2d','moment','sentence_len','index','sentence','duration','phrase'"], per_batch_map=batch_process)
        # for d in dataset:
        #     print('here',d)
        # sampler = make_data_sampler(dataset, is_train)
        # if is_train:
        #     batch_sampler = make_train_data_sampler(dataset, sampler, cfg.SOLVER.BATCH_SIZE)
        # else:
        #     batch_sampler = make_test_data_sampler(dataset, sampler, cfg.SOLVER.BATCH_SIZE)
        # data_loader = torch.utils.data.DataLoader(
        #     dataset,
        #     num_workers=cfg.DATALOADER.NUM_WORKERS,
        #     batch_sampler=batch_sampler,
        #     collate_fn=BatchCollator(),
        # )
        if is_train:
            dataset = dataset.batch(batch_size=cfg.SOLVER.BATCH_SIZE, per_batch_map=batch_process,output_columns=["feature","query","word_len","iou2d","moment","index","sen_len","duration","sentence","phrase"])
        else:
            print('test',cfg.TEST.BATCH_SIZE)
            dataset = dataset.batch(batch_size=cfg.TEST.BATCH_SIZE, per_batch_map=batch_process,output_columns=["feature","query","word_len","iou2d","moment","index","sen_len","duration","sentence","phrase"])
        # for d in dataset:
        #     print(d)
        data_loaders.append(dataset)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
