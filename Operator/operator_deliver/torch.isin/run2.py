import torch
import torch_npu  # 导入华为 NPU 支持

# 检查 NPU 是否可用
if torch.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("NPU is NOT available, falling back to CPU.")

# 定义测试数据并移动到 NPU 上
token_parent_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=device)
unique_token_parent_ids = torch.tensor([1, 3, 5, 7, 9, 11], device=device)

# 取 unique_token_parent_ids 的前半部分
unique_token_parent_ids_half = unique_token_parent_ids[: len(unique_token_parent_ids) // 2]

# 使用 torch.isin 检查 token_parent_ids 中的元素是否存在于 unique_token_parent_ids_half 中
group1_mask = torch.isin(token_parent_ids, unique_token_parent_ids_half)

print("token_parent_ids:", token_parent_ids)
print("unique_token_parent_ids_half:", unique_token_parent_ids_half)
print("group1_mask:", group1_mask)  # 输出: tensor([ True, False,  True, False,  True, False, False, False, False, False], device='npu:0')