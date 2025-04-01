## torch.utils._pytree.tree_map 

功能是为输入的数据结构调用统一的函数进行处理，考虑到line 225:

```
if is_tensor(t) else t, d
```
这个是一个设备无关的if语句选择，用于判断输入数据是否张量，这个函数可以保持为pytorch的进行使用

参考run2.py的代码，运行中基本是纯cpu燃尽

## torch.nn.Parameter 
无关设备，用于创建特殊的张量类型，算是特殊的类构造器，也可以继续沿用pytorch的
需要注意，如果你要把参数放到npu上需要特殊处理迁移到npu上，参考nn.parameter的run1.py的处理方法

***很奇怪啊 只有import torch_npu后.npu函数才可用***

测试文件输出:
```bash
[W308 16:50:13.072766794 compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
weight Parameter containing:
tensor([[ 0.9764,  0.2195, -0.6692],
        [-1.2557, -0.0089,  1.1128]], device='npu:0', requires_grad=True) npu:0
bias Parameter containing:
tensor([ 0.1569, -0.5143], device='npu:0', requires_grad=True) npu:0
Parameter is on NPU!
Gradient of weight: tensor([[ 0.0997, -1.2271,  0.2385],
        [ 0.0997, -1.2271,  0.2385]], device='npu:0')
Gradient of bias: tensor([1., 1.], device='npu:0')
```

考虑alphafold3.py line 1228:
```python
    self.layerscale_output = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.
```
这个`torch.zeros(dim_pairwise)`返回值类型就是Tensor，在后面直接.npu()就可以实现对npu的迁移

## torch.amp.autocast
代码中的使用为
```python
    @typecheck #类型检查装饰不用管
    @autocast("cuda", enabled=False)
    def forward(
```
这个写法是为了屏蔽cuda的混合精度使用,torch_npu有这个可以直接用，参数改成`@autocast(False)`，参看run2.py

同时拿npu的GradScaler试了一下，调参type里面填npu竟然也是可以用的。。。建议直接换成npu的autocast

## torch.svd
这是一个不依赖设备的函数，只需要确保张量在NPU上就可以了。原位置：

```python
        U, S, V = torch.svd(cov_matrix)
```
输入的cov_matrix是一个Tensor@einsum，这个张量的输出位置和输入的参数位置在同一个设备上。参考下面的上下文输入的参数包括了Tensor@Where，Tensor@range，Tensor@Einsum。
```python
        pred_coords = einx.where("b n, b n c, -> b n c", mask, pred_coords, 0.0)
        true_coords = einx.where("b n, b n c, -> b n c", mask, true_coords, 0.0)
        weights = einx.where("b n, b n, -> b n", mask, weights, 0.0)
        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid
        # Compute the weighted covariance matrix
        cov_matrix = einsum(
            weights * true_coords_centered, pred_coords_centered, "b n i, b n j -> b i j"
        )
        # Compute the SVD of the covariance matrix
        U, S, V = torch.svd(cov_matrix)
```
需要注意的是，torch_npu可以直接调用.to或者.npu把张量迁移到npu上，目前要考虑的是，先让CPU处理出cov_matrix后在迁移到npu上再svd，或者前面的全部先迁移到(注意这个特性，你只需要改weights,pre-true的定义位置就可以一直爽了)npu上再einsum再svd(这个我目测人工可能要点时间)。其次就是老登爱谈的性能问题。

参考run3.py，矩阵够大(40960)可以直接吃53g显存，还有这个不要一次塞太大矩阵进去，我估计是npu自己有什么问题，拿到输出之后显存不会第一时间释放，并且不会返回。

## torch.det
输入参数的位置，和上面一样，这个就是和svd差不多
```python
        det = torch.det(einsum(V, U_T, "b i j, b j k -> b i k"))
```
run2.py是高强度作业，做了一点小小的改变，可以迁到别的NPU上面。

## torch.nn.functional.pairwise_distance

```python
# 从4170行开始是获得所有参数的位置，来自一些自写函数创建张量
        pred_coords_transformed = self.express_coordinates_in_frame(pair_pred_coords, pair_pred_frames)
        true_coords_transformed = self.express_coordinates_in_frame(pair_true_coords, pair_true_frames)
        alignment_errors = F.pairwise_distance(pred_coords_transformed, true_coords_transformed, eps = self.eps)
# 函数定义
        (function) def pairwise_distance(
                x1: Tensor,
                x2: Tensor,
                p: _float = 2,
                eps: _float = 0.000001,
                keepdim: _bool = False
        ) -> Tensor

```
这个函数返回一个Tensor，`self.express_coordinates_in_frame`的定义是`class ExpressCoordinatesInFrame(Module):`，来自`alphafold3_pytorch/utils/model_utils.py`的一个自定义类类型。

在run1.py里直接通过引用af3-pytorch的类迁到npu上然后跑。是可以适应原来的上下文参数创建的
run2.py高强度训练，这个东西单吃内存不占AICore怎么个事,吃的是内存带宽

## torch.isin

用于检查一个张量中的元素是否存在于另一个张量中。它返回一个布尔张量，表示每个元素是否存在于目标张量中。

不用看了这是戏子演的，直接张量迁到npu拿着用就行了。run2.py输入类型相同测试

## torch.distributions.Geometric

用于表示几何分布（Geometric Distribution）的类。几何分布是一种离散概率分布，描述在一系列独立伯努利试验中，第一次成功所需的试验次数。

```python
    p: float = 1.0 / 3.0
    geom_distr = torch.distributions.Geometric(torch.tensor([p]))
```

不用看了一眼戏子，直接把Tensor迁移到npu上就行了。run2.py是原生同参数输入调用

## torch.nn.functional.attention

老弟我问啊这个attention在alphafold3.py里面没有调用啊,nn.functional调用为F. 找不到F.attention啊。唯一一个Attention是af3-pt自己写的一个Attention(这个Attention是自定类型，只需要确保输入的时候张量就在npu上)

torch.nn.functional.attention 并不是 PyTorch 中的一个直接存在的函数。PyTorch 提供了 torch.nn.functional.scaled_dot_product_attention 和 torch.nn.MultiheadAttention 来实现注意力机制。以下是对这些函数的详细介绍和用法示例。

?