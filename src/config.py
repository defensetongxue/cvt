from yacs.config import CfgNode as CN
import paddle.nn as nn

_C = CN()

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 256  # 256 # train batch_size for single GPU
_C.DATA.BATCH_STRIDE = 16  # 64 # val batch_size for single GPU
_C.DATA.IN_CHANS = 3
# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'Cvt'
_C.MODEL.NAME = 'Cvt'
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.ACT_LAYER = 'nn.GELU'
_C.MODEL.NORM_LAYER = 'nn.LayerNorm'
_C.MODEL.INIT = 'trunc_norm'

# spec
_C.SPEC = CN()
_C.SPEC.NUM_STAGES = 4
_C.SPEC.PATCH_SIZE = 16
_C.SPEC.PATCH_STRIDE = 16
_C.SPEC.PATCH_PADDING = 0
_C.SPEC.DIM_EMBED = 769
_C.SPEC.DEPTH = 12
_C.SPEC.NUM_HEADS = 12
_C.SPEC.MLP_RATIO = 4
_C.SPEC.QKV_BIAS = False
_C.SPEC.DROP_RATE = 0.
_C.SPEC.ATTN_DROP_RATE = 0.
_C.SPEC.DROP_PATH_RATE = 0.
_C.SPEC.WITH_CLS_TOKEN = False
_C.SPEC.QKV_PROJ_METHOD = 'dw_bn'
_C.SPEC.KERNEL_QKV = 7
_C.SPEC.PADDING_Q = 1
_C.SPEC.PADDING_KV = 1
_C.SPEC.STRIDE_KV = 1
_C.SPEC.STRIDE_Q = 1
_C.SPEC.DIM_EMBED = 12


def get_config():
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    return config
