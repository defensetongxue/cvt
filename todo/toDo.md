## rearrange 简单
`rearrange:func` 和 `Rearrange:nn.Layer` 是基于MIT的开源库enipos的函数，但是不支持`paddle.Tonsor`类型的输入。

目前我的解决办法是采用`PaddleRearrange`和`RearrangeLayer`进行包装，先转化为`np.array`作为`rearrange`的输入，最后将输出转化回来。

更好的解决办法是`fork`enipos，并修改代码使得支持paddle.Tensor 网址： https://github.com/arogozhnikov/einops



## Block 模块审核测试 中等

目前完成api修改，并修改一些基本的函数
继续审核并最好能提供测试




# finish
## attention模块审核测试 中等

目前完成api修改，并修改一些基本的函数

仍然存在问题 1.timm.models.layers.DropPath 可能不等于Paddle.nn.DropOut,需要格外注意

继续审核并最好能提供测试