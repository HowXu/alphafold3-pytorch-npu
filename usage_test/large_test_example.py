# 大量数据 指定使用cuda

import torch
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

import torch_npu

# 检查 GPU 是否可用
if torch_npu.npu.is_available():
    device = torch.device("npu")
    print("NPU is available!")
else:
    device = torch.device("cpu")
    print("No NPU found, using CPU.")

# 创建模型并移动到 GPU
alphafold3 = Alphafold3(
    dim_atom_inputs = 512,
    dim_template_feats = 1024
).to(device)

# 输入数据
seq_len = 512
batch_size = 16

molecule_atom_indices = torch.randint(0, 2, (batch_size, seq_len)).long().to(device)
molecule_atom_lens = torch.full((batch_size, seq_len), 2).long().to(device)

atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

atom_inputs = torch.randn(batch_size, atom_seq_len, 512).to(device)
atompair_inputs = torch.randn(batch_size, atom_seq_len, atom_seq_len, 5).to(device)

additional_molecule_feats = torch.randint(0, 2, (batch_size, seq_len, 5)).to(device)
additional_token_feats = torch.randn(batch_size, seq_len, 33).to(device)
is_molecule_types = torch.randint(0, 2, (batch_size, seq_len, 5)).bool().to(device)
is_molecule_mod = torch.randint(0, 2, (batch_size, seq_len, 4)).bool().to(device)
molecule_ids = torch.randint(0, 32, (batch_size, seq_len)).to(device)

template_feats = torch.randn(batch_size, 16, seq_len, seq_len, 512).to(device)
template_mask = torch.ones((batch_size, 16)).bool().to(device)

msa = torch.randn(batch_size, 64, seq_len, 32).to(device)
msa_mask = torch.ones((batch_size, 64)).bool().to(device)

additional_msa_feats = torch.randn(batch_size, 64, seq_len, 2).to(device)

# 训练数据
atom_pos = torch.randn(batch_size, atom_seq_len, 3).to(device)
distogram_atom_indices = molecule_atom_lens - 1
distance_labels = torch.randint(0, 37, (batch_size, seq_len, seq_len)).to(device)
resolved_labels = torch.randint(0, 2, (batch_size, atom_seq_len)).to(device)

# 偏移索引
distogram_atom_indices += atom_offsets
molecule_atom_indices += atom_offsets

# 训练
loss = alphafold3(
    num_recycling_steps = 8,
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

# 推理
sampled_atom_pos = alphafold3(
    num_recycling_steps = 16,
    num_sample_steps = 64,
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

print(sampled_atom_pos.shape)  # (16, <atom_seqlen>, 3)