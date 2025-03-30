import torch
import time
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

import torch_npu
start_time = time.time()
# Pass ready - HowXu

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108
)# .cuda()

alphafold3 = alphafold3.npu()

# mock inputs
#torch.set_default_tensor_type(torch.cuda.FloatTensor) # This 控制到cuda上 虽然舍弃但是可以用

seq_len = 16

# 这里的输入请控制在npu上
molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long().npu()
molecule_atom_lens = torch.full((2, seq_len), 2).long().npu()

atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens).npu()

atom_inputs = torch.randn(2, atom_seq_len, 77).npu()
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5).npu()

additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5)).npu()
additional_token_feats = torch.randn(2, seq_len, 33).npu()
is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool().npu()
is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool().npu()
molecule_ids = torch.randint(0, 32, (2, seq_len)).npu()

template_feats = torch.randn(2, 2, seq_len, seq_len, 108).npu()
template_mask = torch.ones((2, 2)).bool().npu()

msa = torch.randn(2, 7, seq_len, 32).npu()
msa_mask = torch.ones((2, 7)).bool().npu()

additional_msa_feats = torch.randn(2, 7, seq_len, 2).npu()

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3).npu()

distogram_atom_indices = molecule_atom_lens - 1

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len)).npu()
resolved_labels = torch.randint(0, 2, (2, atom_seq_len)).npu()

# offset indices correctly

distogram_atom_indices += atom_offsets
molecule_atom_indices += atom_offsets

# train
print(alphafold3.device)
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

# after much training ...

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

sampled_atom_pos.shape # (2, <atom_seqlen>, 3)import torch

end_time = time.time()
print(f"Time taken: {end_time - start_time}")
"""
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108
)

# mock inputs

seq_len = 16

molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long()
molecule_atom_lens = torch.full((2, seq_len), 2).long()

atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)

additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5))
additional_token_feats = torch.randn(2, seq_len, 33)
is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool()
is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool()
molecule_ids = torch.randint(0, 32, (2, seq_len))

template_feats = torch.randn(2, 2, seq_len, seq_len, 108)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 32)
msa_mask = torch.ones((2, 7)).bool()

additional_msa_feats = torch.randn(2, 7, seq_len, 2)

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)

distogram_atom_indices = molecule_atom_lens - 1

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
resolved_labels = torch.randint(0, 2, (2, atom_seq_len))

# offset indices correctly

distogram_atom_indices += atom_offsets
molecule_atom_indices += atom_offsets

# train

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

loss.backward() # 调用

# after much training ...

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

sampled_atom_pos.shape # (2, <atom_seqlen>, 3)
"""