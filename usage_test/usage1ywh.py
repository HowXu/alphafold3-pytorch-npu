import torch
import time
from datetime import datetime
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

import torch_npu

def format_time(seconds):
    """格式化时间，将秒数转换为更易读的格式"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes)}分{seconds:.2f}秒"

# 检查并设置设备
device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}")

if torch_npu.npu.is_available():
    print(f"📊 NPU信息:")
    print(f"   - 型号: {torch_npu.npu.get_device_name(0)}")
    print(f"   - 显存总量: {torch_npu.npu.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    print(f"   - 当前显存使用: {torch_npu.npu.memory_allocated() / 1024**2:.1f}MB")

# 记录开始时间
start_time = time.time()
print(f"⏰ 程序开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 初始化 AlphaFold3 模型
model_start = time.time()

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108
).to(device) # 将模型移动到GPU
 
model_time = time.time() - model_start
print(f"🚀 模型初始化完成! (耗时: {format_time(model_time)})")
print(f"📍 模型所在设备: {next(alphafold3.parameters()).device}")

# 模拟输入数据
# ====================
print("\n📥 准备输入数据...")
data_prep_start = time.time()

seq_len = 16  # 序列长度

# 分子原子索引和长度
molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long().to(device)
molecule_atom_lens = torch.full((2, seq_len), 2).long().to(device)

# 计算原子序列长度和偏移量
atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

print(f"   序列长度: {seq_len}")
print(f"   原子序列长度: {atom_seq_len}")

# 生成各种输入特征并移动到GPU
# 原子级别的特征
atom_inputs = torch.randn(2, atom_seq_len, 77).to(device)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5).to(device)

# 分子级别的特征
additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5)).to(device)
additional_token_feats = torch.randn(2, seq_len, 33).to(device)
is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool().to(device)
is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool().to(device)
molecule_ids = torch.randint(0, 32, (2, seq_len)).to(device)

# 模板特征
template_feats = torch.randn(2, 2, seq_len, seq_len, 108).to(device)
template_mask = torch.ones((2, 2)).bool().to(device)

# MSA特征
msa = torch.randn(2, 7, seq_len, 32).to(device)
msa_mask = torch.ones((2, 7)).bool().to(device)
additional_msa_feats = torch.randn(2, 7, seq_len, 2).to(device)

data_prep_time = time.time() - data_prep_start
print(f"📊 输入数据准备完成! (耗时: {format_time(data_prep_time)})")

if torch_npu.npu.is_available():
    print(f"   当前NPU内存使用: {torch_npu.npu.memory_allocated() / 1024**2:.1f}MB")

# 训练所需的标签数据
print("\n🏷️ 准备标签数据...")
atom_pos = torch.randn(2, atom_seq_len, 3).to(device)
distogram_atom_indices = molecule_atom_lens - 1
distance_labels = torch.randint(0, 37, (2, seq_len, seq_len)).to(device)
resolved_labels = torch.randint(0, 2, (2, atom_seq_len)).to(device)

# 调整索引偏移
distogram_atom_indices = distogram_atom_indices.to(device) + atom_offsets
molecule_atom_indices = molecule_atom_indices + atom_offsets

# 训练阶段
train_start = time.time()
print(f"🎯 开始训练... ({datetime.now().strftime('%H:%M:%S')})")

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,
    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    distogram_atom_indices = distogram_atom_indices,
    molecule_atom_indices = molecule_atom_indices,
    distance_labels = distance_labels,
    resolved_labels = resolved_labels
)

loss.backward()
train_time = time.time() - train_start
print(f"📉 训练损失: {loss.item():.4f} (训练耗时: {format_time(train_time)})")

if torch_npu.npu.is_available():
    print(f"   训练后NPU内存使用: {torch_npu.npu.memory_allocated() / 1024**2:.1f}MB")

# 采样预测阶段
predict_start = time.time()
print(f"🔄 开始采样预测... ({datetime.now().strftime('%H:%M:%S')})")

sampled_atom_pos = alphafold3(
    num_recycling_steps = 4,
    num_sample_steps = 16,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,
    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

predict_time = time.time() - predict_start
print(f"✨ 预测完成! 输出形状: {sampled_atom_pos.shape}")
print(f"🔍 预测的原子位置内容:")
print(sampled_atom_pos.cpu().detach().numpy())  # 将结果移回CPU并转换为NumPy数组

# GPU 内存使用情况
if torch_npu.npu.is_available():
    print(f"\n📊 NPU内存使用情况:")
    print(f"   - 已分配: {torch_npu.npu.memory_allocated() / 1024**2:.1f}MB")
    print(f"   - 已缓存: {torch_npu.npu.memory_reserved() / 1024**2:.1f}MB")

# 性能统计
total_time = time.time() - start_time
print("\n⏱️ 性能统计:")
print(f"├── 模型初始化: {format_time(model_time)}")
print(f"├── 数据准备: {format_time(data_prep_time)}")
print(f"├── 训练阶段: {format_time(train_time)}")
print(f"├── 预测阶段: {format_time(predict_time)}")
print(f"└── 总耗时: {format_time(total_time)}")
print(f"🏁 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 清理GPU内存
if torch_npu.npu.is_available():
    torch_npu.npu.empty_cache()
    print("🧹 已清理NPU缓存")