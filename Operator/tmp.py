import torch
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.data import process_pdb

# 1. 初始化模型
model = Alphafold3(
    dim_atom_inputs=77,
    dim_template_feats=44
).cuda()

# 2. 处理输入数据
pdb_path = "1MBN.pdb"
processed_data = process_pdb(pdb_path)

# 3. 转换为Tensor
inputs = {
    k: torch.tensor(v).float().cuda() 
    for k, v in processed_data.items()
}

# 4. 推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# 5. 保存结果
save_as_pdb(outputs["positions"][0].cpu().numpy(), "predicted.pdb")