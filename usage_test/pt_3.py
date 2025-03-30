#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行 AlphaFold 3 PyTorch 实现的蛋白质结构预测
支持多种蛋白质序列，自动计算原子数量，并保存结果
"""

import torch
import numpy as np
from alphafold3_pytorch import Alphafold3, Alphafold3Input
import os
import time
import argparse
from datetime import datetime
import sys
import tempfile  # 导入tempfile模块用于处理临时文件
import torch_npu

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 预定义的蛋白质序列
PREDEFINED_SEQUENCES = {
    "insulin_a": "GIVEQCCTSICSLYQLENYCN",  # 胰岛素A链，21个氨基酸
    "insulin_b": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # 胰岛素B链，30个氨基酸
    "ag": "AG",  # 丙氨酸-甘氨酸二肽，2个氨基酸
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 泛素，76个氨基酸
    "lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"  # 溶菌酶，129个氨基酸
}

def get_atom_counts_per_residue():
    """返回每种氨基酸的非氢原子数量"""
    # 标准氨基酸的非氢原子数量
    atom_counts = {
        'A': 5,  # 丙氨酸 (ALA)
        'R': 11, # 精氨酸 (ARG)
        'N': 8,  # 天冬酰胺 (ASN)
        'D': 8,  # 天冬氨酸 (ASP)
        'C': 6,  # 半胱氨酸 (CYS)
        'Q': 9,  # 谷氨酰胺 (GLN)
        'E': 9,  # 谷氨酸 (GLU)
        'G': 4,  # 甘氨酸 (GLY)
        'H': 10, # 组氨酸 (HIS)
        'I': 8,  # 异亮氨酸 (ILE)
        'L': 8,  # 亮氨酸 (LEU)
        'K': 9,  # 赖氨酸 (LYS)
        'M': 8,  # 甲硫氨酸 (MET)
        'F': 11, # 苯丙氨酸 (PHE)
        'P': 7,  # 脯氨酸 (PRO)
        'S': 6,  # 丝氨酸 (SER)
        'T': 7,  # 苏氨酸 (THR)
        'W': 14, # 色氨酸 (TRP)
        'Y': 12, # 酪氨酸 (TYR)
        'V': 7   # 缬氨酸 (VAL)
    }
    return atom_counts

def generate_mock_atom_positions(sequence, device=None):
    """为给定的氨基酸序列生成模拟的原子位置
    
    参数:
        sequence: 氨基酸序列字符串
        device: 计算设备，如果为None则使用默认设备
    
    返回:
        包含每个氨基酸原子位置的张量列表
    """
    atom_counts = get_atom_counts_per_residue()
    mock_atompos = []
    
    for aa in sequence:
        # 获取该氨基酸的原子数量，默认为5
        num_atoms = atom_counts.get(aa, 5)
        # 为每个原子创建随机位置
        if device is not None:
            mock_atompos.append(torch.randn(num_atoms, 3, device=device))
        else:
            mock_atompos.append(torch.randn(num_atoms, 3))
    
    return mock_atompos

def save_pdb_with_biopython(atom_pos, sequence, filename):
    """使用 BioPython 保存原子坐标为 PDB 文件"""
    try:
        # 确保atom_pos是numpy数组
        if torch.is_tensor(atom_pos):
            atom_pos = atom_pos.cpu().numpy()
        
        from Bio.PDB import PDBParser, PDBIO, StructureBuilder
        from Bio.PDB.Atom import Atom
        from Bio.PDB.Residue import Residue
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Model import Model
        from Bio.PDB.Structure import Structure
        
        # 氨基酸三字母代码映射
        aa_map = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        # 每个氨基酸的主要原子名称
        atom_names = {
            'ALA': ['N', 'CA', 'C', 'O', 'CB'],
            'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
            'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
            'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
            'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
            'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
            'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
            'GLY': ['N', 'CA', 'C', 'O'],
            'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
            'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
            'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
            'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
            'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
            'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
            'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
            'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
            'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
            'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2']
        }
        
        # 创建结构
        structure = Structure("predicted")
        model = Model(0)
        chain = Chain("A")
        
        atom_idx = 0
        for i, aa in enumerate(sequence):
            res_name = aa_map.get(aa, 'UNK')
            residue = Residue((' ', i+1, ' '), res_name, '')
            
            # 获取这个氨基酸的原子名称
            names = atom_names.get(res_name, ['CA'])  # 默认至少有一个 CA 原子
            
            # 为每个原子添加坐标
            for j, name in enumerate(names):
                if atom_idx < len(atom_pos):
                    try:
                        # 获取原子坐标
                        if isinstance(atom_pos, np.ndarray):
                            coord = atom_pos[atom_idx]
                        else:
                            coord = atom_pos[atom_idx].cpu().numpy() if torch.is_tensor(atom_pos[atom_idx]) else atom_pos[atom_idx]
                    except (IndexError, TypeError) as e:
                        print(f"警告: 无法获取原子 {atom_idx} 的坐标: {str(e)}")
                        print(f"atom_pos类型: {type(atom_pos)}, 形状: {atom_pos.shape if hasattr(atom_pos, 'shape') else 'unknown'}")
                        # 使用默认坐标
                        coord = np.zeros(3)
                    
                    atom = Atom(name, coord, 0.0, 1.0, ' ', name, j, element=name[0])
                    residue.add(atom)
                    atom_idx += 1
                else:
                    break
            
            chain.add(residue)
        
        model.add(chain)
        structure.add(model)
        
        # 保存为 PDB 文件
        io = PDBIO()
        io.set_structure(structure)
        io.save(filename)
        return True
    except Exception as e:
        print(f"使用 BioPython 保存 PDB 时出错: {str(e)}")
        return False

def run_alphafold3(sequence, output_dir=None, train_steps=50, device=None, **kwargs):
    """
    运行AlphaFold 3模型进行蛋白质结构预测
    
    参数:
        sequence (str): 蛋白质序列，使用一字母氨基酸代码
        output_dir (str, 可选): 输出目录路径，如果为None则使用系统临时目录
        train_steps (int): 训练步骤数，默认为50
        device (torch.device, 可选): 计算设备，如果为None则自动选择
        print_tensor (bool, 可选): 是否打印tensor的值，默认为False
        
    返回:
        torch.Tensor: 预测的原子位置
    """
    # 获取可选参数
    print_tensor = kwargs.get('print_tensor', False)
    
    # 导入必要的库
    import torch
    import time
    import os
    import tempfile
    from datetime import datetime
    import traceback
    from pathlib import Path

    
    # 获取氨基酸原子数量映射
    atom_counts = get_atom_counts_per_residue()
    
    # 导入Alphafold3相关模块
    try:
        from alphafold3_pytorch import Alphafold3Input
        from alphafold3_pytorch.alphafold3 import Alphafold3
    except ImportError:
        raise ImportError("需要安装alphafold3-pytorch库: pip install alphafold3-pytorch")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    # 设置设备
    if device is None:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device("npu")
        device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
    
    # 生成模拟原子位置
    mock_atompos = generate_mock_atom_positions(sequence, device)
    
    # 创建新的AlphaFold3模型
    print("创建AlphaFold3模型...")
    # 创建模型，参考 run_af3_pt2.py 的参数设置
    model = Alphafold3(
        dim_atom_inputs = 3,
        dim_atompair_inputs = 5,
        atoms_per_window = 27,
        dim_template_feats = 108,
        num_molecule_mods = 0,
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
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    model = model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建训练输入 - 使用 Alphafold3Input 类
    train_alphafold3_input = Alphafold3Input(
        proteins = [sequence],
        atom_pos = mock_atompos
    )
    
    # 创建评估输入
    eval_alphafold3_input = Alphafold3Input(
        proteins = [sequence]
    )
    
    # 执行训练步骤
    print("执行训练步骤...")
    start_time = time.time()
    try:
        # 使用 forward_with_alphafold3_inputs 方法
        loss = model.forward_with_alphafold3_inputs([train_alphafold3_input])
        print(f"训练损失: {loss.item():.4f}")
        
        # 额外的训练步骤
        if train_steps > 0:
            print(f"执行额外的 {train_steps} 次训练步骤...")
            for i in range(train_steps):
                optimizer.zero_grad()
                loss = model.forward_with_alphafold3_inputs([train_alphafold3_input])
                loss.backward()
                optimizer.step()
                print(f"  步骤 {i+1}/{train_steps}, 损失: {loss.item():.4f}")
        
        end_time = time.time()
        print(f"训练完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 执行结构预测
        print("执行结构预测...")
        start_time = time.time()
        
        # 使用模型进行预测
        model.eval()
        with torch.no_grad():
            # 直接进行预测
            sampled_atom_pos = model.forward_with_alphafold3_inputs(eval_alphafold3_input)
        
        end_time = time.time()
        print(f"预测完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 打印结果信息
        print(f"采样原子位置形状: {sampled_atom_pos.shape}")
        
        # 检查预期的原子数量
        expected_atom_count = sum(atom_counts.get(aa, 5) for aa in sequence)
        print(f"预期原子数量: {expected_atom_count}")
        
        # 计算并打印统计信息
        print(f"原子位置平均值: {sampled_atom_pos.mean().item():.4f}")
        print(f"原子位置标准差: {sampled_atom_pos.std().item():.4f}")
        
        # 打印tensor的值
        if print_tensor:
            print("\n预测的原子位置:")
            print(sampled_atom_pos)
        
        # 保存预测结构
        if output_dir is not None:
            print("保存预测结构...")
            try:
                # 尝试创建输出目录
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成文件名
                if len(sequence) <= 10:
                    seq_str = sequence
                else:
                    seq_str = f"{sequence[:5]}...{sequence[-5:]}"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdb_file = os.path.join(output_dir, f"{seq_str}_{timestamp}.pdb")
                
                # 尝试将预测结果转换为PDB格式并保存
                save_result = save_pdb_with_biopython(sampled_atom_pos[0], sequence, pdb_file)
                
                if save_result:
                    print(f"结构已成功保存到: {pdb_file}")
                    if output_dir == tempfile.gettempdir():
                        print(f"注意: 这是系统临时目录中的文件，可能会被系统自动清理")
                        print(f"可以使用以下命令复制到其他位置: cp \"{pdb_file}\" 目标路径/")
                else:
                    # 如果保存失败，尝试保存到临时目录
                    temp_dir = tempfile.gettempdir()
                    fallback_pdb_file = os.path.join(temp_dir, f"{seq_str}_{timestamp}.pdb")
                    print(f"无法保存到 {pdb_file}，尝试保存到临时目录: {fallback_pdb_file}")
                    
                    if save_pdb_with_biopython(sampled_atom_pos[0], sequence, fallback_pdb_file):
                        print(f"结构已成功保存到: {fallback_pdb_file}")
                        print(f"注意: 这是系统临时目录中的文件，可能会被系统自动清理")
                        print(f"可以使用以下命令复制到其他位置: cp \"{fallback_pdb_file}\" 目标路径/")
                    else:
                        print("无法保存结构文件，可能没有写入权限")
            except Exception as e:
                # 捕获任何保存过程中的错误
                print(f"保存结构时出错: {str(e)}")
                print("尝试保存到系统临时目录...")
                try:
                    temp_dir = tempfile.gettempdir()
                    fallback_pdb_file = os.path.join(temp_dir, f"{sequence}_{timestamp}.pdb")
                    if save_pdb_with_biopython(sampled_atom_pos[0], sequence, fallback_pdb_file):
                        print(f"结构已成功保存到系统临时目录: {fallback_pdb_file}")
                        print(f"注意: 这是临时文件，可能会被系统自动清理")
                        print(f"可以使用以下命令复制到其他位置: cp \"{fallback_pdb_file}\" 目标路径/")
                    else:
                        print("无法保存结构文件，请检查权限")
                except Exception as fallback_error:
                    print(f"备用保存也失败了: {str(fallback_error)}")
                    print("请检查系统是否有足够的磁盘空间和写入权限")
        
        print("预测完成!")
        return sampled_atom_pos
    
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        
        # 尝试获取更多调试信息
        print("\n尝试获取更多调试信息...")
        try:
            print(f"原始 mock_atompos 列表长度: {len(mock_atompos)}")
            for i, (aa, pos) in enumerate(zip(sequence, mock_atompos)):
                print(f"  位置 {i} (氨基酸 {aa}): 形状 {pos.shape}")
            
            print("\n如果出现形状不匹配错误，请检查以下信息:")
            print("1. 确保每个氨基酸的原子数量与模型期望的一致")
            print("2. 可能需要调整 generate_mock_atom_positions 函数中的原子数量")
            print("3. 考虑使用更简单的序列进行测试，如 'AG'")
        except Exception as debug_error:
            print(f"调试信息获取失败: {str(debug_error)}")
        
        # 重新抛出异常
        raise

def print_atom_count_helper_code():
    """打印辅助程序的代码，让用户可以手动创建"""
    print("\n由于权限问题，无法自动创建辅助程序文件。")
    print("请使用已创建的 af3_atom_count_helper.py 文件，或查看该文件获取更多功能。")

def run_example():
    """运行一个简单的示例，使用预定义的序列"""
    print("运行示例: 使用 'AG' 二肽序列")
    sequence = "AG"  # 丙氨酸-甘氨酸二肽，简单序列用于测试
    
    # 使用系统临时目录，确保有写入权限
    output_dir = tempfile.gettempdir()
    print(f"输出将保存到系统临时目录: {output_dir}")
    
    train_steps = 50  # 增加训练步骤以获得更好的结果
    print(f"使用 {train_steps} 个训练步骤")
    
    # 检测设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行预测
    run_alphafold3(sequence, output_dir, train_steps, device, print_tensor=True)

def main():
    # 默认输出目录设置为系统临时目录
    default_output_dir = tempfile.gettempdir()
    
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        # 没有参数时运行示例
        run_example()
        return
    
    # 有参数时使用命令行参数运行
    parser = argparse.ArgumentParser(description="运行AlphaFold 3预测蛋白质结构")
    parser.add_argument("sequence", help="蛋白质序列", nargs="?", default=None)
    parser.add_argument("-o", "--output", help="输出目录", default=default_output_dir)
    parser.add_argument("-t", "--train_steps", type=int, default=50, help="训练步骤数")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--example", action="store_true", help="运行示例")
    parser.add_argument("--no-print-tensor", dest="print_tensor", action="store_false", help="不打印tensor数据")
    parser.set_defaults(print_tensor=True)
    
    args = parser.parse_args()
    
    # 如果指定了--example选项或没有提供序列，运行示例
    if args.example or args.sequence is None:
        run_example()
        return
    
    # 设置设备
    device = torch.device('cpu') if args.cpu else torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
    # evice = torch.device('npu')
    
    # 运行AlphaFold 3
    run_alphafold3(args.sequence, args.output, args.train_steps, device, print_tensor=args.print_tensor)

if __name__ == "__main__":
    main() 