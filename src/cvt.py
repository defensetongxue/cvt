import paddle
import paddle.nn as nn

from einops import rearrange

from collections.abc import Iterable
from numpy import repeat

from collections import OrderedDict

import logging
import os

from config import get_config 

import numpy as np

def PaddleRearrange(tensor:paddle.Tensor, pattern: str, **axes_lengths) -> paddle.Tensor:
    x=np.array(tensor)
    return paddle.to_tensor(rearrange(x,pattern,**axes_lengths))
class RearrangeLayer(nn.Layer):
    def __init__(self, pattern):
        super().__init__()
        self.pattern=pattern
    def forword(self,x:paddle.Tensor):
        return PaddleRearrange(x,self.pattern)


# From PyTorch internals
"""对repeat进行封装，让代码更加健壮"""
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):#如果已经是转换后的值，直接返回，不需要再做转换操作
            return x
        return tuple(repeat(x, n))

    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(paddle.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.sigmoid(1.702 * x)

class Mlp(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.XavierUniform()) #default in pp: xavier
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(std=1e-6)) #default in pp: zero
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class ConvEmbed(nn.Layer):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)#把patch初始化为一个正方形,这里是(7,7)

        self.patch_size = patch_size
        self.proj = nn.Conv2D(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape#B个图片H*W的大小 C个通道(example：W==3:红黄蓝)
        x = PaddleRearrange(x, 'b c h w -> b (h w) c')#对每个图片进行嵌入，相当于对每个图片线性的堆叠
        if self.norm:
            x = self.norm(x)
        x = PaddleRearrange(x, 'b (h w) c -> b c h w', h=H, w=W)#把x回归原来的形状

        return x

class Attention(nn.Layer):
    """ Attention module
    Attributes:
        dim_in: numebr of input dim
        dim_out: number of output dum
        num_heads: 
        qkv_bias=False
        attn_drop=0.,
        proj_drop=0.
        method='dw_bn' generate projection method 
        kernel_size=3  conv kernel size 
        stride_kv=1 calculat k,v with conv , with paramer stride 
        stride_q=1 calculat qwith conv , with paramer stride ,this stride can be different from stride_kv=1
        padding_kv=1  calculat k,v with conv , with paramer paddding
        padding_q=1 calculat q with conv , with paramer paddding
        with_cls_token=True ,if label is given
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        #init to save the pararm
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        # calculate q,k,v with conv

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        # init parameters of q,k,v

        self.proj_q = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)

        # init project other parameters

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(
                ( nn.Conv2D(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias_attr=False,
                    groups=dim_in
                )),
                (nn.BatchNorm2D(dim_in)),
                ( RearrangeLayer('b c h w -> b (h w) c')),
            )
        elif method == 'avg':
            proj = nn.Sequential(
                ( nn.AvgPool2D(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                (RearrangeLayer('b c h w -> b (h w) c')),
            )
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token: # spilt token from x
            cls_token, x = paddle.split(x, [1, h*w], 1)

        x =  PaddleRearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q =  PaddleRearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k =  PaddleRearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v =  PaddleRearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = paddle.gather(cls_token, q, dim=1)
            k = paddle.gather(cls_token, k, dim=1)
            v = paddle.gather(cls_token, v, dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):#if not generate q,k,v with Linear param
            q, k, v = self.forward_conv(x, h, w)
        #now q,k,v is b (h w) c
        q =  PaddleRearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)#先扩宽token的维度，然后再实现mult-head，最后的结构是’b,h,t,d‘
        k =  PaddleRearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v =  PaddleRearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = paddle.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale # 先按照axis=3乘，后*scale，实现q*k/sqort(d_k),
        attn = nn.functional.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.einsum('bhlt,bhtv->bhlv', [attn, v]) # 将attention得到概率值与value信息值相乘得到结果，结构是，b,h,t,d
        x =  PaddleRearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x#b,t,(h,d)

class Block(nn.Layer):
    ''' 
    每一个Block都是
    token -> multihead attention ( reshape token to a grap) ->Mlp->token
    '''
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )
        if drop_path>0. :
            self.drop_path=nn.Dropout(drop_path)
        else:
            self.drop_path=nn.Identity()
        
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            dim_in,
            mlp_ratio,
            dropout=drop
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            trun_init=nn.initializer.TruncatedNormal(std=0.02)
            trun_init(self.cls_token)
        else:
            self.cls_token = None
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.LayerList(blocks)

        

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trun_init=nn.initializer.TruncatedNormal(std=0.02)
            trun_init(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                zeros=nn.initializer.Constant(0.)
                zeros(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros=nn.initializer.Constant(0.)
            zeros(m.bias)
            ones=nn.initializer.Constant(1)
            ones(m.weight)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            xavier_init=nn.initializer.XavierNormal()
            xavier_init(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                zeros=nn.initializer.Constant(0.)
            zeros(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros=nn.initializer.Constant(0.)
            zeros(m.bias)
            ones=nn.initializer.Constant(1)
            ones(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = paddle.gather(cls_tokens, x, dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H*W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens

class ConvolutionalVisionTransformer(nn.Layer):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec.NUM_STAGES
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec.PATCH_SIZE,
                'patch_stride': spec.PATCH_STRIDE ,
                'patch_padding': spec.PATCH_PADDING ,
                'embed_dim': spec.DIM_EMBED ,
                'depth': spec.DEPTH ,
                'num_heads': spec.NUM_HEADS ,
                'mlp_ratio': spec.MLP_RATIO ,
                'qkv_bias': spec.QKV_BIAS ,
                'drop_rate': spec.DROP_RATE ,
                'attn_drop_rate': spec.ATTN_DROP_RATE ,
                'drop_path_rate': spec.DROP_PATH_RATE,
                'with_cls_token': spec.WITH_CLS_TOKEN,
                'method': spec.QKV_PROJ_METHOD,
                'kernel_size': spec.KERNEL_QKV ,
                'padding_q': spec.PADDING_Q ,
                'padding_kv': spec.PADDING_KV ,
                'stride_kv': spec.STRIDE_KV ,
                'stride_q': spec.STRIDE_Q ,
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec.DIM_EMBED

        dim_embed = spec.DIM_EMBED
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec.WITH_CLS_TOKEN

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_init=nn.initializer.TruncatedNormal(std=0.02)
        trunc_init(self.head.weight)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = paddle.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = paddle.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = paddle.to_tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)
    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = paddle.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x



def generate_model(config):
    if config.MODEL.ACT_LAYER == 'nn.GELU':
        act_layer=nn.GELU
    if config.MODEL.NORM_LAYER == 'nn.LayerNorm':
        norm_layer=nn.LayerNorm
    model=ConvolutionalVisionTransformer(
                 in_chans=config.DATA.IN_CHANS,
                 num_classes=config.MODEL.NUM_CLASSES,
                 act_layer=act_layer ,
                 norm_layer=norm_layer,
                 init=config.MODEL.INIT ,
                 spec=config.SPEC)
    return model

config=get_config()
x=generate_model(config)