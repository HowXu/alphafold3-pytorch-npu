import torch
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 创建两个随机矩阵并移动到 NPU
A = torch.randn(5, 3).to(device)  # 5x3 矩阵
B = torch.randn(3, 4).to(device)  # 3x4 矩阵

# 使用 torch.einsum 进行矩阵乘法 这里是直接参考了AlphaFold3的输入参数形式
C = torch.einsum('ij,jk->ik', A, B)  # 结果是一个 5x4 矩阵
print("C is on device:", C.device)  # 检查C的device 输出值预计为C is on device: npu:0
# 对结果进行奇异值分解
U, S, V = torch.svd(C)

print("U:", U)
print("S:", S)
print("V:", V)

# 验证分解结果
C_reconstructed = U @ torch.diag(S) @ V.t()
print("Reconstructed C:", C_reconstructed)

if torch.npu.is_available():
    torch_npu.npu.empty_cache()