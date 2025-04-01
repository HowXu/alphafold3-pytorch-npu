import torch
import torch.nn as nn
from torch_npu.npu.amp import autocast
#from torch import typecheck  # 假设这是类型检查装饰器

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1024, 1024)

    #@typecheck  # 类型检查
    @autocast(False)  # 在 NPU 上禁用混合精度
    def forward(self, x):
        # 模型的前向传播逻辑
        return self.fc(x)

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 创建模型实例并移动到 NPU
model = SimpleModel().to(device)

# 创建随机输入数据
inputs = torch.randn(128, 1024, device=device)

# 运行前向传播
outputs = model(inputs)
print(outputs)