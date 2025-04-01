#梯度测试和反向传播测试

import torch
import torch_npu  # 导入华为 NPU 支持 
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 定义一个 Parameter，并将其放在 NPU 上
        self.weight = nn.Parameter(torch.randn(2, 3).npu())  # 将参数放在 NPU 上
        self.bias = nn.Parameter(torch.randn(2).npu())       # 将参数放在 NPU 上

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

# 创建模型实例
model = SimpleModel()

# 打印模型参数
for name, param in model.named_parameters():
    print(name, param, param.device)

# 检查参数是否在 NPU 上
if str(model.weight.device).startswith('npu'):
    # This is expected output
    print("Parameter is on NPU!")
else:
    print("Parameter is NOT on NPU.")
# 创建一个输入张量，并将其放在 NPU 上
x = torch.randn(1, 3).npu()

# 前向传播
output = model(x)

# 计算损失
loss = output.sum()

# 反向传播
loss.backward()

# 打印梯度
for name, param in model.named_parameters():
    print(f"Gradient of {name}: {param.grad}")

#输出 测试正常
"""
[W308 16:43:14.835376109 compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
weight Parameter containing:
tensor([[0.5412, 2.0807, 1.0644],
        [1.3196, 1.1131, 2.9119]], device='npu:0', requires_grad=True) npu:0
bias Parameter containing:
tensor([-0.4281, -0.7717], device='npu:0', requires_grad=True) npu:0
Parameter is on NPU!
Gradient of weight: tensor([[-0.4650,  0.5750,  1.2951],
        [-0.4650,  0.5750,  1.2951]], device='npu:0')
Gradient of bias: tensor([1., 1.], device='npu:0')
"""