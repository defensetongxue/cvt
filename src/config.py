from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 256 #256 # train batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 8 #64 # val batch_size for single GPU
_C.DATA.DATA_PATH = '/dataset/imagenet/' # path to dataset
_C.DATA.DATASET = 'imagenet2012' # dataset name
_C.DATA.IMAGE_SIZE = 224 # input image size: 224 for pretrain, 384 for finetune
_C.DATA.CROP_PCT = 0.875 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 2 # number of data loading threads 

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'ViT'
_C.MODEL.NAME = 'ViT'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.1
_C.MODEL.DROPPATH = 0.1
_C.MODEL.ATTENTION_DROPOUT = 0.1

# transformer settings
_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.PATCH_SIZE = 32
_C.MODEL.TRANS.EMBED_DIM = 768
_C.MODEL.TRANS.MLP_RATIO= 4.0
_C.MODEL.TRANS.NUM_HEADS = 12
_C.MODEL.TRANS.DEPTH = 12
_C.MODEL.TRANS.QKV_BIAS = True

# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 3 #34 # ~ 10k steps for 4096 batch size
_C.TRAIN.WEIGHT_DECAY = 0.05 #0.3 # 0.0 for finetune
_C.TRAIN.BASE_LR = 0.001 #0.003 for pretrain # 0.03 for finetune
_C.TRAIN.WARMUP_START_LR = 1e-6 #0.0
_C.TRAIN.END_LR = 5e-4
_C.TRAIN.GRAD_CLIP = 1.0
_C.TRAIN.ACCUM_ITER = 2 #1

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'warmupcosine'
_C.TRAIN.LR_SCHEDULER.MILESTONES = "30, 60, 90" # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1 # only used in StepLRScheduler

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)  # for adamW
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# misc
_C.SAVE = "./output"
_C.TAG = "default"
_C.SAVE_FREQ = 10 # freq to save chpt
_C.REPORT_FREQ = 100 # freq to logging info
_C.VALIDATE_FREQ = 100 # freq to do validation
_C.SEED = 0
_C.EVAL = False # run evaluation only
_C.AMP = False # mix precision training
_C.LOCAL_RANK = 0
_C.NGPUS = -1




def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    return config