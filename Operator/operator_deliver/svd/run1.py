# 单独检查svd的支持情况
import torch
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 创建一个随机矩阵并移动到 NPU
A = torch.randn(5, 3).to(device)  # 5x3 矩阵

# 计算奇异值分解
U, S, V = torch.svd(A)

print("U:", U)
print("S:", S)
print("V:", V)

# 验证分解结果
A_reconstructed = U @ torch.diag(S) @ V.t()
print("Reconstructed A:", A_reconstructed)

if torch.npu.is_available():
    torch_npu.npu.empty_cache()
# 依然是可以直接跑到NPU上的