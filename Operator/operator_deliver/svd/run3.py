import torch
import torch_npu  # 导入华为 NPU 支持
import time

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 定义矩阵大小
matrix_size = 4096  # 增大矩阵大小以提高计算量
batch_size = 1     # 增加批量大小以进一步增加计算量

# 创建随机矩阵并移动到 NPU
A = torch.randn(matrix_size, matrix_size, device=device)
B = torch.randn(matrix_size, matrix_size, device=device)

# 测试 torch.einsum 和 torch.svd 的性能
start_time = time.time()

for i in range(batch_size):
    # 使用 torch.einsum 进行矩阵乘法
    C = torch.einsum('ij,jk->ik', A, B)  # 结果是一个 matrix_size x matrix_size 矩阵
    #print("C is on device:", C.device)  # 检查C的device
    # 对结果进行奇异值分解
    U, S, V = torch.svd(C)
    
    # 打印进度
    if (i + 1) % 1 == 0:
        print(f"Step {i + 1} completed.")

# 计算总时间
total_time = time.time() - start_time
print(f"Total time: {total_time:.2f} seconds")