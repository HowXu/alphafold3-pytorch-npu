import torch
from torch.distributions import Geometric
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 创建几何分布对象并移动到 NPU 上
probs = torch.tensor(0.3, device=device)
geom_dist = Geometric(probs=probs)

# 采样 10 个值
samples = geom_dist.sample((10,))
print("Samples:", samples)  # 输出: tensor([...], device='npu:0')

# 计算 k=5 的对数概率
log_prob = geom_dist.log_prob(torch.tensor(5.0, device=device))
print("Log probability of k=5:", log_prob)  # 输出: tensor([...], device='npu:0')