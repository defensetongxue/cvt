import numpy as np
import paddle
import paddle.nn as nn

from einops import rearrange


from collections.abc import Iterable
from numpy import repeat
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

        B, C, H, W = x.shape#B个图片 C*H的大小 W个通道(example：W==3:红黄蓝)
        x = rearrange(x, 'b c h w -> b (h w) c')#对每个图片进行嵌入，相当于对每个图片线性的堆叠
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)#把x回归原来的形状

        return x
x=np.random.rand(0,30,10,90,90,3)#0~30 (10,90,90,3)的随机数
x=x.astype(np.float)
ConvEmbed()(paddle.to_tensor(x))#问：即使我这里转化成float，但是依然会然为我input的类型是double
"""报错：
input and filter data type should be consistent,
 but received input data type is double and filter
  type is float
  [Hint: Expected input_data_type == filter_data_type, but received input_data_type:6 != filter_data_type:5.] 
  (at C:\home\workspace\Paddle_release\paddle/fluid/operators/conv_op.cc:211)




"""