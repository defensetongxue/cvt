## 报错信息：
TypeError: assignment to parameter 'bias' should be of type Parameter or None, but got 'Constant'
```python
    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            m.weight=nn.initializer.TruncatedNormal(std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                m.bias=nn.initializer.Constant(0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            m.bias=nn.initializer.Constant(0.)  ##报错 
            m.weight=nn.initializer.Constant(1.0)

```
## 文档：
https://www.paddlepaddle.org.cn/documentation/docs/zh/2.1/api/paddle/nn/initializer/Constant_cn.html#constant

## 我的思考
这里文档中可以用nn.initializer.Constant直接初始化参数，我这里单独拿出来初始化却不行，这里可能可以换成,但具体维度也要考虑。不知道init.constant具体如何实现此处的代码。
```python
m.bias=paddle.to_tensor(nn.zeros(shape[embed_dim]))
```
源码的实现是:
```python
nn.init.constant_(m.bias, 0)
```

## 所在类完整代码
```python
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
            self.cls_token=nn.initializer.TruncatedNormal(std=0.02)
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
            m.weight=nn.initializer.TruncatedNormal(std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                m.bias=nn.initializer.Constant(0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            m.bias=nn.initializer.Constant(0.)
            m.weight=nn.initializer.Constant(1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                m.bias=nn.initializer.Constant(0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            m.bias=nn.initializer.Constant(0.)
            m.weight=nn.initializer.Constant(1.0)

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
```
