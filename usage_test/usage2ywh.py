import torch
from alphafold3_pytorch import Alphafold3, Alphafold3Input
import time
from datetime import datetime

import torch_npu

# 格式化时间的函数
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

contrived_protein = 'AG'

# 将数据移动到指定设备
mock_atompos = [
    # 这个输入可以不移动
    torch.randn(5, 3).to(device),   # alanine has 5 non-hydrogen atoms
    torch.randn(4, 3).to(device)    # glycine has 4 non-hydrogen atoms
]

# 准备训练输入
train_alphafold3_input = Alphafold3Input(
    proteins=[contrived_protein],
    atom_pos=mock_atompos
)

eval_alphafold3_input = Alphafold3Input(
    proteins=[contrived_protein]
)

# 模型初始化并移动到GPU
model_start = time.time()
alphafold3 = Alphafold3(
    dim_atom_inputs=3,
    dim_atompair_inputs=5,
    atoms_per_window=27,
    dim_template_feats=108,
    num_molecule_mods=0,
    confidence_head_kwargs=dict(
        pairformer_depth=1
    ),
    template_embedder_kwargs=dict(
        pairformer_stack_depth=1
    ),
    msa_module_kwargs=dict(
        depth=1
    ),
    pairformer_stack=dict(
        depth=2
    ),
    diffusion_module_kwargs=dict(
        atom_encoder_depth=1,
        token_transformer_depth=1,
        atom_decoder_depth=1,
    )
).to(device)  # 将模型移动到NPU

model_time = time.time() - model_start
print(f"🚀 模型初始化完成! (耗时: {format_time(model_time)})")

# 打印模型所在设备
print(f"📍 模型所在设备: {next(alphafold3.parameters()).device}")

# 训练阶段
train_start = time.time()
loss = alphafold3.forward_with_alphafold3_inputs([train_alphafold3_input])
loss.backward()
train_time = time.time() - train_start
print(f"🎯 训练完成! 训练损失: {loss.item():.4f} (训练耗时: {format_time(train_time)})")

# 采样阶段
sample_start = time.time()
alphafold3.eval()
sampled_atom_pos = alphafold3.forward_with_alphafold3_inputs(eval_alphafold3_input)
sample_time = time.time() - sample_start
print(f"🔄 采样完成! 输出形状: {sampled_atom_pos.shape} (采样耗时: {format_time(sample_time)})")

# 打印具体的 Tensor 内容
print("🔍 预测的原子位置内容:")
print(sampled_atom_pos.cpu().detach().numpy())  # 将结果移回CPU并转换为NumPy数组

# GPU 内存使用情况
if torch_npu.npu.is_available():
    print(f"\n📊 NPU内存使用情况:")
    print(f"   - 已分配: {torch_npu.npu.memory_allocated() / 1024**2:.1f}MB")
    print(f"   - 已缓存: {torch_npu.npu.memory_reserved() / 1024**2:.1f}MB")

# 总耗时
total_time = time.time() - start_time
print(f"⏱️ 总耗时: {format_time(total_time)}")
print(f"🏁 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 清理NPU内存
if torch_npu.npu.is_available():
    torch_npu.npu.empty_cache()
    print("🧹 已清理NPU缓存")