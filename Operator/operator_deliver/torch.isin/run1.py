import torch
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 定义输入张量和目标张量并移动到 NPU 上
elements = torch.tensor([1, 2, 3, 4, 5], device=device)
test_elements = torch.tensor([2, 4, 6], device=device)

# 检查 elements 中的元素是否存在于 test_elements 中
result = torch.isin(elements, test_elements)

print("elements:", elements)
print("test_elements:", test_elements)
print("result:", result)  # 输出: tensor([False,  True, False,  True, False], device='npu:0')