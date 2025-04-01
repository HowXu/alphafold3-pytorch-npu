import torch
import torch.nn as nn
import torch_npu  # 导入华为 NPU 支持
import time

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 定义一个计算密集型模型
class ComputeIntensiveModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ComputeIntensiveModel, self).__init__()
        # 定义可训练参数
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size, device=device)) # 和用.npu一样的
        self.weight2 = nn.Parameter(torch.randn(output_size, hidden_size, device=device))
        self.bias1 = nn.Parameter(torch.randn(hidden_size, device=device))
        self.bias2 = nn.Parameter(torch.randn(output_size, device=device))

    def forward(self, x):
        # 大规模矩阵乘法
        x = torch.matmul(x, self.weight1.t()) + self.bias1
        x = torch.relu(x)  # 激活函数
        x = torch.matmul(x, self.weight2.t()) + self.bias2
        return x

# 定义模型参数
power = 8 #直接按倍数扔进去造就能看NPU有没有在动

input_size = 4096*power  # 输入维度
hidden_size = 8192*power  # 隐藏层维度
output_size = 4096*power  # 输出维度
batch_size = 128*power  # 批量大小
num_steps = 100*power  # 迭代次数

# 创建模型实例
model = ComputeIntensiveModel(input_size, hidden_size, output_size).to(device)

# 创建随机输入数据
inputs = torch.randn(batch_size, input_size, device=device)

# 测试前向传播和反向传播
start_time = time.time()
for i in range(num_steps):
    # 前向传播
    outputs = model(inputs)
    
    # 计算损失（假设是均方误差）
    target = torch.randn(batch_size, output_size, device=device)
    loss = torch.mean((outputs - target) ** 2)
    
    # 反向传播
    loss.backward()
    
    # 打印损失
    if (i + 1) % 10 == 0:
        print(f"Step {i + 1}, Loss: {loss.item()}")

# 计算总时间
total_time = time.time() - start_time
print(f"Total time: {total_time:.2f} seconds")

"""
(1) 模型定义

    ComputeIntensiveModel 是一个计算密集型模型，包含两个全连接层。

    使用 nn.Parameter 定义模型的权重和偏置，并将它们直接放在 NPU 上（通过 device=device）。

(2) 输入数据

    输入数据是一个随机生成的张量，大小为 (batch_size, input_size)，并直接放在 NPU 上。

(3) 计算任务

    前向传播：进行大规模的矩阵乘法（torch.matmul）和激活函数计算。

    损失计算：使用均方误差（MSE）作为损失函数。

    反向传播：调用 loss.backward() 计算梯度。

(4) 资源占用

    通过设置较大的 input_size、hidden_size 和 batch_size，确保计算任务足够密集，能够充分利用 NPU 资源。

(5) 性能监控

    使用 time.time() 计算总运行时间，评估 NPU 的计算性能。

    
(1) 显式使用 .npu()

    在定义模型参数时，使用 .npu() 将张量移动到 NPU 上：
    python
    复制

    self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size).npu())

    在创建输入数据时，使用 .npu() 将张量移动到 NPU 上：
    python
    复制

    inputs = torch.randn(batch_size, input_size).npu()

(2) 计算任务

    前向传播：进行大规模的矩阵乘法（torch.matmul）和激活函数计算。

    损失计算：使用均方误差（MSE）作为损失函数。

    反向传播：调用 loss.backward() 计算梯度。

(3) 资源占用

    通过设置较大的 input_size、hidden_size 和 batch_size，确保计算任务足够密集，能够充分利用 NPU 资源。
"""