# Alphafold3 Pytorch on NPU

based on [AlphaFold3-pytorch](https://github.com/lucidrains/alphafold3-pytorch)

这个分支修改了部分关键算子，使AlphaFold3-Pytorch在使用昇腾PyTorch的情况下依靠NPU进行训练，推理，预测。

原版README见[Here](./README_ORIGIN.md)

# 算子改动情况

## torch.autocast/torch_npu.npu.amp.autocast

在原生Pytorch中，这个算子不支持NPU，因此只能更换为torch_npu的算子，这也间接导致本分支只能在cpu/npu环境进行使用。

## torch.cdist/alphafold3.npu.cdist_npu

这个算子直接不支持NPU并且拥有相对较多的调用，因此进行了重写，见[npu.py](./alphafold3_pytorch/npu.py)

## torch.svd & torch.det

这两个算子在NPU运算下精度损失很大，本分支尝试进行重写但是未能彻底解决精度问题，考虑其占比较小，这一部分固定为CPU调用。

# 用法

clone本项目后，执行:

```shell
pip install .
```

## 框架测试
目前还在开发测试阶段，你可以使用下面命令进行完整的测试:

```shell
python pt_6.py --run-complete-pipeline --sequence MKTVRQ --epochs 3 --learning-rate 0.0001
```

更多测试案例见[测试脚本例6](./usage_test/pt_6.py)。需要指出，分支并没有将底层所有类和函数的默认张量迁移到NPU上，[原有的测试例](./tests/)并不适用。

## 用例

参考[原版README](./README_ORIGIN.md)的两个用例，你只需要进行一些修改:

1. 将传入类构造和前向传播函数的张量迁移到npu上(可以使用[环境统一的迁移函数](./alphafold3_pytorch/npu.py)tensor_to_npu)。

2. 将构造的AlphaFold3对象迁移到NPU上(这里使用.to_device函数或者.npu函数)

在多卡情况下，你需要按照实际情况修改tensor_to_npu函数，控制.to_device函数及.npu的参数。

### 例1
```py
import torch
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

from alphafold3_pytorch.npu import tensor_to_npu #导入
import torch_npu

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108
)

alphafold3 = alphafold3.npu() # 迁移核心对象

# mock inputs

seq_len = 16

molecule_atom_indices = tensor_to_npu(torch.randint(0, 2, (2, seq_len)).long())
molecule_atom_lens = tensor_to_npu(torch.full((2, seq_len), 2).long())

atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

atom_inputs = tensor_to_npu(torch.randn(2, atom_seq_len, 77))
atompair_inputs = tensor_to_npu(torch.randn(2, atom_seq_len, atom_seq_len, 5))

additional_molecule_feats = tensor_to_npu(torch.randint(0, 2, (2, seq_len, 5)))
additional_token_feats = tensor_to_npu(torch.randn(2, seq_len, 33))
is_molecule_types = tensor_to_npu(torch.randint(0, 2, (2, seq_len, 5)).bool())
is_molecule_mod = tensor_to_npu(torch.randint(0, 2, (2, seq_len, 4)).bool())
molecule_ids = tensor_to_npu(torch.randint(0, 32, (2, seq_len)))

template_feats = tensor_to_npu(torch.randn(2, 2, seq_len, seq_len, 108))
template_mask = tensor_to_npu(torch.ones((2, 2)).bool())

msa = tensor_to_npu(torch.randn(2, 7, seq_len, 32))
msa_mask = tensor_to_npu(torch.ones((2, 7)).bool())

additional_msa_feats = tensor_to_npu(torch.randn(2, 7, seq_len, 2))

# required for training, but omitted on inference

atom_pos = tensor_to_npu(torch.randn(2, atom_seq_len, 3))

distogram_atom_indices = molecule_atom_lens - 1

distance_labels = tensor_to_npu(torch.randint(0, 37, (2, seq_len, seq_len)))
resolved_labels = tensor_to_npu(torch.randint(0, 2, (2, atom_seq_len)))

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

sampled_atom_pos.shape # (2, <atom_seqlen>, 3)
```

### 例2

```py
import torch
from alphafold3_pytorch import Alphafold3, Alphafold3Input

from alphafold3_pytorch.npu import tensor_to_npu # 导入

contrived_protein = 'AG'

mock_atompos = [
    torch.randn(5, 3),   # alanine has 5 non-hydrogen atoms
    torch.randn(4, 3)    # glycine has 4 non-hydrogen atoms
]

train_alphafold3_input = Alphafold3Input(
    proteins = [contrived_protein],
    atom_pos = mock_atompos
)

eval_alphafold3_input = Alphafold3Input(
    proteins = [contrived_protein]
)

# training

alphafold3 = Alphafold3(
    dim_atom_inputs = 3,
    dim_atompair_inputs = 5,
    atoms_per_window = 27,
    dim_template_feats = 108,
    num_molecule_mods = 0,
    confidence_head_kwargs = dict(
        pairformer_depth = 1
    ),
    template_embedder_kwargs = dict(
        pairformer_stack_depth = 1
    ),
    msa_module_kwargs = dict(
        depth = 1
    ),
    pairformer_stack = dict(
        depth = 2
    ),
    diffusion_module_kwargs = dict(
        atom_encoder_depth = 1,
        token_transformer_depth = 1,
        atom_decoder_depth = 1,
    )
)

alphafold3 = alphafold3.npu() # 迁移

loss = alphafold3.forward_with_alphafold3_inputs([train_alphafold3_input])
loss.backward()

# sampling

alphafold3.eval()
sampled_atom_pos = alphafold3.forward_with_alphafold3_inputs(eval_alphafold3_input)

assert sampled_atom_pos.shape == (1, (5 + 4), 3)
```