## 1-class:LayerNorm
layerNorm类为什么返回时一个type，之后写到再详细看看
## 2-class:attention
timm.models.layers.DropPath 可能不等于Paddle.nn.DropOut,需要格外注意
## 3-class:attention
删除了compute_macs：并知道这个函数做什么的的