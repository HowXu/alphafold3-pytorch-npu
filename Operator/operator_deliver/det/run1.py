import torch
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 创建一个随机方阵并移动到 NPU 上
A = torch.randn(4096, 4096).to(device)

# 计算行列式
det_A = torch.det(A)

print("A:", A)
print("det(A):", det_A)