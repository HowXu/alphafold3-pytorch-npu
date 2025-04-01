import torch
from torch.utils._pytree import tree_map
import torch_npu  # 导入华为 NPU 支持
import time

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 定义一个函数，将张量移动到 NPU 上
def move_to_npu(tensor):
    if torch.is_tensor(tensor):
        return tensor.to(device)  # 使用 .to(device) 将张量移动到 NPU
    return tensor

# 定义一个计算密集型函数
def compute_intensive_op(tensor):
    # 大规模矩阵乘法
    return torch.matmul(tensor, tensor.t())


xi_huang_zhuang = 16384*8
# 定义一个嵌套的数据结构，包含更大的张量
nested_data = {
    'a': [torch.randn(xi_huang_zhuang, xi_huang_zhuang), torch.randn(xi_huang_zhuang, xi_huang_zhuang)],
    'b': (torch.randn(xi_huang_zhuang, xi_huang_zhuang), torch.randn(xi_huang_zhuang, xi_huang_zhuang)),
    'c': {'d': torch.randn(xi_huang_zhuang, xi_huang_zhuang), 'e': torch.randn(xi_huang_zhuang, xi_huang_zhuang)}
}

# 使用 tree_map 将数据结构中的所有张量移动到 NPU
print("Moving data to NPU...")
npu_data = tree_map(move_to_npu, nested_data)

# 验证张量是否在 NPU 上
def check_device(data):
    if torch.is_tensor(data):
        print(f"Tensor is on device: {data.device}")
    return data

print("Checking devices...")
tree_map(check_device, npu_data)

# 测试 tree_map 的计算性能
start_time = time.time()
num_steps = 10  # 迭代次数

print("Starting computation...")
for i in range(num_steps):
    # 使用 tree_map 对嵌套数据结构中的每个张量执行计算密集型操作
    result = tree_map(compute_intensive_op, npu_data)
    
    # 打印进度
    if (i + 1) % 1 == 0:
        print(f"Step {i + 1} completed.")

# 计算总时间
total_time = time.time() - start_time
print(f"Total time: {total_time:.2f} seconds")

"""
(1) tree_map 的使用

    tree_map 用于递归地遍历嵌套数据结构（如字典、列表、元组等），并对每个张量执行指定的操作。

    在本例中，tree_map 被用于：

        将嵌套数据结构中的所有张量移动到 NPU 上（通过 move_to_npu 函数）。

        对嵌套数据结构中的每个张量执行计算密集型操作（通过 compute_intensive_op 函数）。

(2) 计算密集型操作

    compute_intensive_op 函数执行大规模的矩阵乘法（torch.matmul），这是一个计算密集型操作，能够充分利用 NPU 的计算资源。

(3) 嵌套数据结构

    nested_data 是一个嵌套的数据结构，包含多个大小为 (1024, 1024) 的随机张量。

    通过 tree_map，我们可以方便地对这些张量进行操作。

(4) 性能监控

    使用 time.time() 计算总运行时间，评估 NPU 的计算性能。
"""