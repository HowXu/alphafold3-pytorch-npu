# 基本功能测试

import torch
import torch.nn as nn
import torch_npu  # 导入华为 NPU 支持

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