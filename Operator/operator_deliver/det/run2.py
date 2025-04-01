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

# 定义矩阵大小和批量大小
matrix_size = 4096  # 增大矩阵大小以提高计算量
batch_size = 10     # 增加批量大小以进一步增加计算量
device = torch.device("npu:2") #这里选设备
# 创建随机矩阵并移动到 NPU 上
A = torch.randn(matrix_size, matrix_size, device=device)

# 测试 torch.det 的性能
start_time = time.time()

for i in range(batch_size):
    # 计算行列式
    det_A = torch.det(A)
    
    # 打印进度
    if (i + 1) % 1 == 0:
        print(f"Step {i + 1} completed.")

# 计算总时间
total_time = time.time() - start_time
print(f"Total time: {total_time:.2f} seconds")

# 释放 NPU 缓存
if torch.npu.is_available():
    torch.npu.empty_cache()

# 确保程序正常退出
print("Program finished.")