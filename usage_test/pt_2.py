import torch
import time
from alphafold3_pytorch import Alphafold3, Alphafold3Input
import torch_npu
start_time = time.time()
contrived_protein = 'AG'
# contrived_protein = 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGLEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH'

# 这里的输入不需要控制在npu上 ?
mock_atompos = [
    torch.randn(5, 3),   # alanine has 5 non-hydrogen atoms
    torch.randn(4, 3)     # glycine has 4 non-hydrogen atoms
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

# to npu
alphafold3 = alphafold3.npu()
print(alphafold3.device)
loss = alphafold3.forward_with_alphafold3_inputs([train_alphafold3_input])
loss.backward()

# sampling

alphafold3.eval()
sampled_atom_pos = alphafold3.forward_with_alphafold3_inputs(eval_alphafold3_input)

assert sampled_atom_pos.shape == (1, (5 + 4), 3)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")