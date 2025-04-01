import torch
import torch.nn as nn
import torch.optim as optim
# from torch_npu.amp import autocast, GradScaler  # 导入 NPU 的 AMP 工具 
# from pytorch 1.10这个导入是不能用的 可以直接用torch库的
from torch.amp import autocast
from torch_npu.npu.amp import GradScaler
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例并移动到 NPU
model = SimpleModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建 GradScaler 实例 
scaler = GradScaler() #这里换成npu的GradScaler

# 创建随机输入数据和目标数据
inputs = torch.randn(128, 1024, device=device)
targets = torch.randn(128, 1024, device=device)

# 训练循环
for i in range(1000):  # 假设训练 1000 个步骤
    optimizer.zero_grad()
    
    # 在 autocast 上下文中运行前向传播 这里要加参数
    with autocast(device_type="npu"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 使用 GradScaler 进行梯度缩放和反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # 打印损失
    print(f"Step {i + 1}, Loss: {loss.item()}")