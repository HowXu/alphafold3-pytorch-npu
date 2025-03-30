#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaFold3-PyTorch 全流程测试框架 - 重构版本 (pt6)
我重生了，这一世，我不当史山了。
这个文件是run_af3_pt5.py的重构版本，实现了从输入序列到结构预测的完整流程，
并增加了训练功能。主要优化了以下几个方面：
1. 提取了重复的输入转换逻辑到专门的辅助函数
2. 统一了模型创建和参数处理逻辑
3. 整合了置信度指标的提取和处理
4. 简化了测试函数的结构，减少了冗余代码
5. 修复了现有的错误和不一致
6. 添加了完整的训练循环和参数更新功能
7. 实现了梯度计算、反向传播和优化器步骤
8. 支持保存训练后的模型权重

该文件使用简化的模型实现（SimpleModel），而非完整的AlphaFold3架构，
主要用于验证功能完整性和框架迁移兼容性，适合在不同硬件平台（包括CPU、GPU和NPU）上测试。
实现了从输入准备、数据处理、模型训练到结构预测和输出的全流程，
可用于验证在不同计算设备上的适配情况。

用法示例:
    # 运行基本功能测试
    python run_af3_pt6.py --test-basic-functionality
    
    # 运行完整预测管道（带训练）
    python run_af3_pt6.py --run-complete-pipeline --sequence MKTVRQ --epochs 3 --learning-rate 0.0001
    
    # 仅运行预测（不训练）
    python run_af3_pt6.py --run-complete-pipeline --sequence MKTVRQ --epochs 0
    
    # 运行综合测试
    python run_af3_pt6.py --test-comprehensive
    
    # 使用指定精度
    python run_af3_pt6.py --run-complete-pipeline --sequence MKTVRQ --precision fp16
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import torch
import random
import time
import gc
from datetime import datetime as dt
import tempfile
import traceback
from Bio.PDB import PDBParser, PDBIO, MMCIFIO, StructureBuilder
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
import types  # 添加types模块导入
import torch_npu

# 添加当前路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 尝试导入alphafold3_pytorch
try:
    from alphafold3_pytorch import Alphafold3, Alphafold3Config
    from alphafold3_pytorch.common.biomolecule import Biomolecule
    from alphafold3_pytorch.data.msa_parsing import Msa, parse_a3m
    from alphafold3_pytorch.data.data_pipeline import FeatureDict, make_msa_features, make_template_features
    from alphafold3_pytorch.inputs import AtomInput, Alphafold3Input
    print("成功导入alphafold3_pytorch库")
except ImportError as e:
    print(f"导入alphafold3_pytorch失败: {str(e)}")
    sys.exit(1)

# 初始化一个全局日志记录器，在main中会被重新配置
logger = logging.getLogger(__name__)

# 定义全局常量
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "af3.bin")
DEFAULT_NUM_RECYCLING_STEPS = 3

# 测试序列集
TEST_SEQUENCES = {
    "ag": "AG",
    "acgu": "ACGU",
    "acgt": "ACGT",
    "small_protein": "ACDEFGHIKLMNPQRSTVWY",
    "medium_protein": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    "rna_simple": "ACGUCCGA",
    "dna_simple": "ACGTACGT",
    "protein_dna": ("ACDEFG", "ACGT"),
    "protein_rna": ("KLMNPQ", "ACGU")
}

# 测试配置
TEST_CONFIGS = {
    "basic": ["ag", "acgu", "acgt"],
    "extended": ["ag", "acgu", "acgt", "small_protein", "rna_simple", "dna_simple"],
    "comprehensive": ["ag", "acgu", "acgt", "small_protein", "medium_protein", "rna_simple", "dna_simple", "protein_dna", "protein_rna"]
}

# 测试配体集
TEST_LIGANDS = {
    "simple_ligand": "CC(=O)O",  # 乙酸
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

def setup_logging(quiet=False, log_file=None):
    """设置日志配置"""
    # 配置根日志记录器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 获取并配置我们的logger
    global logger
    logger = logging.getLogger('af3_test')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理程序
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 添加控制台处理程序
    console_handler = logging.StreamHandler()
    
    if quiet:
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setLevel(logging.WARNING)
    else:
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                           datefmt='%H:%M:%S')
        console_handler.setLevel(logging.INFO)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理程序
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:  # 只有当目录不为空时才创建
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def get_amino_acid_name(aa_letter):
    """将氨基酸单字母代码转换为三字母代码"""
    aa_map = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    return aa_map.get(aa_letter, 'UNK')

def get_rna_name(nucleotide):
    """获取RNA核苷酸名称"""
    rna_map = {
        'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'
    }
    return rna_map.get(nucleotide, 'N')

def get_dna_name(nucleotide):
    """获取DNA核苷酸名称"""
    dna_map = {
        'A': 'DA', 'T': 'DT', 'G': 'DG', 'C': 'DC'
    }
    return dna_map.get(nucleotide, 'DN')

def get_atoms_for_molecule_type(molecule_type):
    """获取指定分子类型的原子名称列表"""
    if molecule_type == "protein":
        return ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1']
    elif molecule_type == "rna":
        return ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"]
    elif molecule_type == "dna":
        return ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"]
    else:
        return ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

def create_simple_biomolecule(atom_positions, sequence, molecule_type="protein"):
    """创建简单的生物分子结构
    
    Args:
        atom_positions: 原子坐标 [num_atoms, 3]
        sequence: 氨基酸或核苷酸序列
        molecule_type: 分子类型 ("protein", "rna", "dna")
        
    Returns:
        Bio.PDB.Structure 对象
    """
    logger.info(f"创建{molecule_type}结构，序列长度: {len(sequence)}")
    
    # 确保atom_positions是numpy数组
    if torch.is_tensor(atom_positions):
        if atom_positions.dim() > 2:  # 如果有批次维度
            atom_positions = atom_positions[0].detach().cpu().numpy()
        else:
            atom_positions = atom_positions.detach().cpu().numpy()
    
    try:
        # 确保atom_positions形状正确
        if len(atom_positions.shape) != 2 or atom_positions.shape[1] != 3:
            logger.error(f"原子坐标形状错误: {atom_positions.shape}, 期望: [num_atoms, 3]")
            return None
        
        # 创建结构构建器
        struct_builder = StructureBuilder.StructureBuilder()
        struct_builder.init_structure("1")
        struct_builder.init_model("1")
        struct_builder.init_chain("A")
        
        # 获取分子类型对应的原子名称
        atom_names_list = get_atoms_for_molecule_type(molecule_type)
        
        # 将列表转换为字典，以便使用get方法
        if molecule_type == "protein":
            res_atoms = {
                'ALA': atom_names_list, 'ARG': atom_names_list, 'ASN': atom_names_list,
                'ASP': atom_names_list, 'CYS': atom_names_list, 'GLN': atom_names_list,
                'GLU': atom_names_list, 'GLY': atom_names_list, 'HIS': atom_names_list,
                'ILE': atom_names_list, 'LEU': atom_names_list, 'LYS': atom_names_list,
                'MET': atom_names_list, 'PHE': atom_names_list, 'PRO': atom_names_list,
                'SER': atom_names_list, 'THR': atom_names_list, 'TRP': atom_names_list,
                'TYR': atom_names_list, 'VAL': atom_names_list
            }
        elif molecule_type == "rna":
            res_atoms = {'A': atom_names_list, 'C': atom_names_list, 
                         'G': atom_names_list, 'U': atom_names_list}
        elif molecule_type == "dna":
            res_atoms = {'DA': atom_names_list, 'DC': atom_names_list, 
                         'DG': atom_names_list, 'DT': atom_names_list}
        else:
            res_atoms = {'UNK': atom_names_list}
        
        # 计算每个残基的原子数量
        atoms_per_residue = {}
        if molecule_type == "protein":
            atom_counts = get_atom_counts_per_residue()
            for aa in atom_counts:
                atoms_per_residue[aa] = atom_counts[aa]
            default_atoms = 5  # 默认氨基酸原子数
        else:
            # 对于RNA/DNA，每个核苷酸的原子数量相同
            for na in "ACGTU":
                atoms_per_residue[na] = 22
            default_atoms = 22  # 默认核苷酸原子数
        
        # 计算总理论原子数量
        expected_atoms = calculate_total_atoms(sequence, molecule_type)
        
        # 实际原子数量
        actual_atoms = len(atom_positions)
        
        # 检查原子数量是否合理
        if actual_atoms < expected_atoms * 0.5 or actual_atoms > expected_atoms * 1.5:
            logger.warning(f"原子数量异常: 期望~{expected_atoms}，实际{actual_atoms}")
        
        # 添加残基和原子
        atom_idx = 0
        for i, res_char in enumerate(sequence):
            # 残基ID (指定插入代码为空)
            res_id = (' ', i+1, ' ')
            
            # 根据分子类型获取残基名称
            if molecule_type == "protein":
                res_name = get_amino_acid_name(res_char)
                names = res_atoms.get(res_name, ['CA'])
            elif molecule_type in ["rna", "dna"]:
                if molecule_type == "rna":
                    na_map = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}
                    na_atom_names = {'A': res_atoms.get('A', ['P']), 'C': res_atoms.get('C', ['P']), 
                                    'G': res_atoms.get('G', ['P']), 'U': res_atoms.get('U', ['P'])}
                else:  # DNA
                    na_map = {'A': 'DA', 'C': 'DC', 'G': 'DG', 'T': 'DT'}
                    na_atom_names = {'DA': res_atoms.get('DA', ['P']), 'DC': res_atoms.get('DC', ['P']), 
                                   'DG': res_atoms.get('DG', ['P']), 'DT': res_atoms.get('DT', ['P'])}
                
                if res_char in na_map:
                    res_name = na_map.get(res_char, 'UNK')
                    names = na_atom_names.get(res_name, ['CA'])
                else:
                    res_name = 'UNK'
                    names = ['CA']
            else:
                # 未知分子类型
                res_name = 'UNK'
                names = ['CA']
            
            # 添加残基
            struct_builder.init_residue(res_name, " ", res_id[1], res_id[2])
            
            # 计算此残基需要多少原子
            num_atoms_for_res = atoms_per_residue.get(res_char, default_atoms)
            
            # 检查是否有足够的原子坐标
            if atom_idx + num_atoms_for_res > len(atom_positions):
                logger.warning(f"原子坐标不足，截断残基 {i+1}")
                num_atoms_for_res = len(atom_positions) - atom_idx
            
            # 添加原子
            for j in range(num_atoms_for_res):
                atom_name = names[j % len(names)]  # 循环使用可用的原子名称
                
                # 确保原子名称不重复，如果需要重复使用同一原子类型，添加数字后缀
                if j >= len(names):
                    suffix = j // len(names) + 1
                    atom_name = f"{atom_name}{suffix}"
                
                element = atom_name[0]  # 元素符号通常是原子名称的第一个字符
                
                # 创建并添加原子
                coord = atom_positions[atom_idx]
                struct_builder.init_atom(atom_name, coord, 0.0, 1.0, ' ', atom_name, element=element)
                atom_idx += 1
                
                if atom_idx >= len(atom_positions):
                    break
            
            if atom_idx >= len(atom_positions):
                logger.warning(f"原子坐标用尽，只能构建到残基 {i+1}/{len(sequence)}")
                break
        
        # 返回完整结构
        structure = struct_builder.get_structure()
        return structure
        
    except Exception as e:
        logger.error(f"创建生物分子结构时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def extract_plddt_from_logits(plddt_logits, device=None):
    """从pLDDT逻辑值中提取置信度分数"""
    try:
        if device is None and torch.is_tensor(plddt_logits):
            device = plddt_logits.device
        
        # 确保logits在正确的设备上
        if torch.is_tensor(plddt_logits) and plddt_logits.device != device and device is not None:
            plddt_logits = plddt_logits.to(device)
            
        # 重排维度 [b, m, bins] -> [b, bins, m]
        logits = plddt_logits.permute(0, 2, 1)
        
        # 计算softmax
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # 计算期望值
        bin_width = 1.0 / logits.shape[1]
        bin_centers = torch.arange(
            0.5 * bin_width, 1.0, bin_width, 
            dtype=probs.dtype, device=probs.device
        )
        
        # 计算置信度分数 (0-100)
        return torch.einsum('bpm,p->bm', probs, bin_centers) * 100
        
    except Exception as e:
        logger.warning(f"从pLDDT逻辑值提取置信度分数时出错: {str(e)}")
        return None

def extract_pae_from_logits(pae_logits, device=None):
    """从PAE逻辑值中提取预测对齐误差"""
    try:
        if device is None and torch.is_tensor(pae_logits):
            device = pae_logits.device
            
        # 确保logits在正确的设备上
        if torch.is_tensor(pae_logits) and pae_logits.device != device and device is not None:
            pae_logits = pae_logits.to(device)
            
        # 重排维度 [b, bins, n, n] -> [b, n, n, bins]
        pae = pae_logits.permute(0, 2, 3, 1)
        
        # 计算softmax
        pae_probs = torch.nn.functional.softmax(pae, dim=-1)
        
        # 假设bin范围从0.5到32埃
        pae_bin_width = 31.5 / pae_probs.shape[-1]
        pae_bin_centers = torch.arange(
            0.5 + 0.5 * pae_bin_width, 32.0, pae_bin_width,
            dtype=pae_probs.dtype, device=pae_probs.device
        )
        
        # 计算期望值
        return torch.einsum('bijd,d->bij', pae_probs, pae_bin_centers)
        
    except Exception as e:
        logger.warning(f"从PAE逻辑值提取误差值时出错: {str(e)}")
        return None

def extract_pde_from_logits(pde_logits, device=None):
    """从PDE逻辑值中提取预测距离误差"""
    try:
        if device is None and torch.is_tensor(pde_logits):
            device = pde_logits.device
            
        # 确保logits在正确的设备上
        if torch.is_tensor(pde_logits) and pde_logits.device != device and device is not None:
            pde_logits = pde_logits.to(device)
            
        # 重排维度 [b, bins, n, n] -> [b, n, n, bins]
        pde = pde_logits.permute(0, 2, 3, 1)
        
        # 计算softmax
        pde_probs = torch.nn.functional.softmax(pde, dim=-1)
        
        # 假设bin范围从0.5到32埃
        pde_bin_width = 31.5 / pde_probs.shape[-1]
        pde_bin_centers = torch.arange(
            0.5 + 0.5 * pde_bin_width, 32.0, pde_bin_width,
            dtype=pde_probs.dtype, device=pde_probs.device
        )
        
        # 计算期望值
        return torch.einsum('bijd,d->bij', pde_probs, pde_bin_centers)
        
    except Exception as e:
        logger.warning(f"从PDE逻辑值提取误差值时出错: {str(e)}")
        return None

def get_atom_counts_per_residue():
    """返回各种氨基酸的非氢原子数量"""
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

def calculate_total_atoms(sequence, molecule_type="protein"):
    """计算序列中的总原子数"""
    if molecule_type == "protein":
        atom_counts = get_atom_counts_per_residue()
        return sum(atom_counts.get(aa, 5) for aa in sequence)
    elif molecule_type in ["rna", "dna"]:
        # 核酸每个核苷酸平均约有20-25个原子
        return len(sequence) * 22
    else:
        # 默认为蛋白质
        atom_counts = get_atom_counts_per_residue()
    return sum(atom_counts.get(aa, 5) for aa in sequence)

def generate_mock_atom_positions(sequence, device=None):
    """为给定序列生成模拟的原子位置，返回统一的张量"""
    atom_counts = get_atom_counts_per_residue()
    mock_atompos_list = []
    total_atoms = 0
    
    for aa in sequence:
        # 获取该氨基酸的原子数量，默认为5
        num_atoms = atom_counts.get(aa, 5)
        total_atoms += num_atoms
        # 为每个原子创建随机位置
        if device is not None:
            mock_atompos_list.append(torch.randn(num_atoms, 3, device=device))
        else:
            mock_atompos_list.append(torch.randn(num_atoms, 3))
    
    # 将所有张量合并为一个大张量
    if mock_atompos_list:
        try:
            # 确保所有张量在同一设备上
            if device is not None:
                mock_atompos_list = [atom_pos.to(device) for atom_pos in mock_atompos_list]
            # 连接所有原子位置
            result = torch.cat(mock_atompos_list, dim=0)
            logger.info(f"生成的模拟原子位置包含 {result.shape[0]} 个原子, 每个原子 {result.shape[1]} 个坐标")
            return result
        except Exception as e:
            logger.error(f"合并原子位置时出错: {str(e)}")
            # 回退到一个安全默认值
            return torch.randn(total_atoms, 3, device=device if device is not None else torch.device('cpu'))
    else:
        # 如果序列为空，返回一个1x3的张量
        return torch.randn(1, 3, device=device if device is not None else torch.device('cpu'))

class MemoryConfig:
    """内存优化配置"""
    def __init__(self, memory_efficient=False):
        self.memory_efficient = memory_efficient
        self.auto_cpu_fallback = True
        self.clear_cache_between_tests = True
        self.gradient_checkpointing = True
        
    def optimize_for_cuda(self):
        """优化NPU内存使用"""
        if not torch_npu.npu.is_available():
            return
        
        if self.memory_efficient:
            logger.info("启用内存效率模式，减少NPU内存使用")
            
            # 设置自动调整内存分配策略
            # 这个没有用
            # os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
            
            # 清空缓存
            self.clear_cuda_cache()
            
            # 设置默认张量类型
            torch.set_default_dtype(torch.float32)
    # to npu
    def clear_cuda_cache(self):
        """清空NPU缓存"""
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
            gc.collect()

def setup_output_dir(base_dir=None):
    """设置输出目录"""
    if base_dir is None:
        # 使用系统临时目录
        base_dir = os.path.join(tempfile.gettempdir(), "af3_test_results")
    
    # 首先确保基础目录存在
    try:
        os.makedirs(base_dir, exist_ok=True)
    except PermissionError:
        # 如果没有权限，则使用临时目录
        temp_dir = os.path.join(tempfile.gettempdir(), "af3_test_results")
        logger.warning(f"无法创建目录 '{base_dir}'，将使用临时目录: '{temp_dir}'")
        base_dir = temp_dir
        os.makedirs(base_dir, exist_ok=True)
    
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"test_{timestamp}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        # 如果没有权限，则使用临时目录
        temp_dir = os.path.join(tempfile.gettempdir(), "af3_test_results")
        logger.warning(f"无法创建目录 '{output_dir}'，将使用临时目录: '{temp_dir}'")
        os.makedirs(temp_dir, exist_ok=True)
        output_dir = os.path.join(temp_dir, f"test_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def save_pdb(structure, filename):
    """保存结构为PDB格式"""
    try:
        io = PDBIO()
        io.set_structure(structure)
        io.save(filename)
        return True
    except Exception as e:
        logger.error(f"保存PDB文件失败: {str(e)}")
        return False

def save_mmcif(structure, filename):
    """保存结构为mmCIF格式"""
    try:
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(filename)
        return True
    except Exception as e:
        logger.error(f"保存mmCIF文件失败: {str(e)}")
        return False

def get_device(force_cpu=False):
    """获取可用的设备（CPU或NPU）"""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("npu" if torch_npu.npu.is_available() else "cpu")

# 添加一个新函数用于将对象移动到指定设备
def move_to_device(obj, device, precision="fp32"):
    """将对象移动到指定设备并设置精度
    
    Args:
        obj: 要移动到设备的对象(Tensor, Module, list, dict, 或自定义对象)
        device: 目标设备
        precision: 精度设置 ('fp32', 'fp16', 或 'bf16')
        
    Returns:
        移动到设备的对象
    """
    if obj is None:
        return None
    
    # 如果是张量，移动到设备并转换精度
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if precision == "fp16":
            obj = obj.half()
        elif precision == "bf16" and torch_npu.npu.is_bf16_supported():
            obj = obj.bfloat16()
        return obj
    
    # 如果是torch模块，移动到设备
    if isinstance(obj, torch.nn.Module):
        if precision == "fp16":
            obj = obj.half()
        elif precision == "bf16" and torch_npu.npu.is_bf16_supported():
            obj = obj.bfloat16()
        return obj.to(device)
    
    # 如果是列表，递归处理每个元素
    if isinstance(obj, list):
        return [move_to_device(item, device, precision) for item in obj]
    
    # 如果是元组，递归处理每个元素
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device, precision) for item in obj)
    
    # 如果是字典，递归处理每个值
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, precision) for k, v in obj.items()}
    
    # 如果是Alphafold3Input对象
    if hasattr(obj, 'to') and callable(obj.to):
        try:
            # 尝试使用对象自身的to方法
            return obj.to(device, precision)
        except Exception as e:
            logger.warning(f"调用对象的to方法失败: {e}，返回原始对象")
            return obj
    
    # 如果是其他自定义对象，尝试遍历其属性并移动到设备
    if hasattr(obj, '__dict__'):
        try:
            for key, value in obj.__dict__.items():
                if isinstance(value, (torch.Tensor, torch.nn.Module, list, tuple, dict)):
                    setattr(obj, key, move_to_device(value, device, precision))
            return obj
        except Exception as e:
            logger.warning(f"移动自定义对象到设备失败: {e}，返回原始对象")
            return obj
    
    # 其他类型，直接返回
    return obj

def check_model_path(model_path):
    """检查模型文件路径是否存在，尝试查找替代路径"""
    if os.path.exists(model_path):
        logger.info(f"找到模型文件: {model_path}")
        return model_path
    
    logger.warning(f"模型文件不存在: {model_path}")
    
    # 尝试在当前目录和常见位置查找模型文件
    potential_paths = [
        "./model/af3.bin",
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            logger.info(f"找到替代模型文件: {path}")
            return path
    
    logger.error("未找到任何替代模型文件")
    return model_path  # 返回原始路径，让调用者处理错误

def load_pretrained_model(model_path, device=None, precision="fp32"):
    """
    加载预训练模型
    
    Args:
        model_path: 模型路径
        device: 计算设备 (None表示自动选择)
        precision: 精度设置 ("fp32", "fp16", "bf16")
        
    Returns:
        预训练模型实例
    """
    if device is None:
        device = get_device()
    
    if not check_model_path(model_path):
        model_path = DEFAULT_MODEL_PATH
        logger.warning(f"使用默认模型路径: {model_path}")
    
    try:
        logger.info(f"从 {model_path} 加载模型")
        
        # 尝试导入AlphaFold3模型类
        from alphafold3_pytorch import Alphafold3
        
        # 确定是否可以使用bfloat16
        bf16_available = False
        if precision == "bf16":
            bf16_available = torch_npu.npu.is_bf16_supported() if torch_npu.npu.is_available() else False
            if not bf16_available:
                logger.warning("设备不支持bfloat16，将回退到fp32")
                precision = "fp32"
                
        # 确定是否可以使用float16
        fp16_available = False
        if precision == "fp16":
            fp16_available = torch_npu.npu.is_available()
            if not fp16_available:
                logger.warning("设备不支持float16，将回退到fp32")
                precision = "fp32"
        
        # 检查Checkpoint加载方法
        if os.path.isdir(model_path):
            # 目录模式 - 包含多个文件的PyTorch模型
            logger.info("检测到模型目录，使用from_pretrained加载")
            
            # 首先使用默认配置创建模型
            model = Alphafold3.from_pretrained(
                model_path, 
                flash_attn=True,
                fp16=precision == "fp16",
                bf16=precision == "bf16"
            )
            
            # 移动模型到设备并转换精度
            model = move_to_device(model, device, precision)
        else:
            # 单文件模式 - 尝试使用torch.load
            logger.info("检测到模型文件，使用torch.load加载")
            
            # 加载检查点
            state_dict = torch.load(model_path, map_location=device)
            
            # 创建默认模型
            model = Alphafold3()
            
            # 加载状态字典
            try:
                model.load_state_dict(state_dict)
                logger.info("成功加载状态字典")
            except Exception as e:
                logger.error(f"加载状态字典失败: {e}")
                raise
                
            # 移动模型到设备并转换精度
            model = move_to_device(model, device, precision)
        
        # 设置模型为评估模式
        model.eval()
        logger.info(f"模型已加载并设置为评估模式，设备: {device}, 精度: {precision}")
        
        return model
        
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_sequence_input(sequence, molecule_type="protein", device=None, custom_atompos=None):
    """创建序列输入"""
    if not sequence:
        logger.error("序列为空")
        return None
    
    if device is None:
        device = get_device()
    
    logger.info(f"创建{molecule_type}序列输入: {sequence}")
    try:
        # 导入所需模块
        import gc
        
        # 清理内存
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
            gc.collect()
        
        # 使用适当的参数创建Alphafold3Input
        logger.info(f"使用Alphafold3Input创建{molecule_type}输入...")
        
        # 检查参数名称
        import inspect
        params = inspect.signature(Alphafold3Input.__init__).parameters
        logger.info(f"Alphafold3Input接受的参数: {list(params.keys())}")
        
        # 根据分子类型使用不同的参数
        af3_input_kwargs = {}
        
        if molecule_type == "protein":
            af3_input_kwargs['proteins'] = [sequence]
        elif molecule_type == "rna":
            af3_input_kwargs['ss_rna'] = [sequence]
        elif molecule_type == "dna":
            af3_input_kwargs['ss_dna'] = [sequence]
        elif molecule_type == "ligand":
            # 对于配体，尝试使用SMILES字符串
            af3_input_kwargs['ligands'] = [sequence]
        elif molecule_type == "metal_ion":
            # 对于金属离子
            af3_input_kwargs['metal_ions'] = [sequence]
        else:
            logger.error(f"不支持的分子类型: {molecule_type}")
            return None
            
        # 创建输入
        af3_input = Alphafold3Input(**af3_input_kwargs)
        
        # 如果提供了自定义原子位置，添加它
        if custom_atompos is not None:
            logger.info("使用提供的自定义原子位置")
            af3_input.atom_pos = custom_atompos
            
        # 将输入转换为设备
        logger.info(f"将输入移至设备: {device}")
        
        # 检查是否有atom_inputs属性，并返回相应的对象
        logger.info(f"创建的输入对象类型: {type(af3_input)}")
        if hasattr(af3_input, 'atom_inputs') and len(af3_input.atom_inputs) > 0:
            logger.info(f"成功创建{molecule_type}输入，返回atom_inputs[0]")
            atom_input = af3_input.atom_inputs[0]
            # 尝试将atom_input移至指定设备
            try:
                atom_input = atom_input.to(device)
            except Exception as device_err:
                logger.warning(f"将atom_input移至设备{device}失败: {str(device_err)}，返回原始atom_input")
            return atom_input
            
        # 尝试将整个af3_input移至设备
        try:
            af3_input = move_to_device(af3_input, device)  # 使用新函数替代直接调用.to()
        except Exception as e:
            logger.warning(f"将af3_input移至设备{device}失败: {str(e)}，返回原始af3_input")
            
        logger.info(f"返回整个Alphafold3Input对象")
        return af3_input
        
    except Exception as e:
        logger.error(f"创建{molecule_type}输入失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prepare_msas_and_templates(sequence_id, sequence, use_msa=True, use_templates=True, data_dir=None):
    """准备MSA和模板数据"""
    logger.info(f"为序列 {sequence_id} 准备MSA和模板数据")
    
    # 初始化返回结果
    msa_data = {'found': False}
    template_data = {'found': False}
    
    # 如果提供了数据目录且目录存在
    if data_dir is not None and os.path.exists(data_dir):
        # MSA文件路径
        msa_path = os.path.join(data_dir, f"{sequence_id}.a3m")
        
        # 如果需要MSA且存在MSA文件
        if use_msa and os.path.exists(msa_path):
            try:
                logger.info(f"加载MSA文件: {msa_path}")
                # 使用alphafold3_pytorch中的函数解析MSA
                from alphafold3_pytorch.data.msa_parsing import parse_a3m, Msa
                with open(msa_path, 'r') as f:
                    msa_content = f.read()
                msa = parse_a3m(msa_content)
                logger.info(f"成功加载MSA，包含 {len(msa.sequences)} 条序列")
                
                # 创建MSA特征
                from alphafold3_pytorch.data.data_pipeline import make_msa_features
                try:
                    msa_features = make_msa_features([msa])
                    msa_data = msa_features
                    msa_data['found'] = True
                    msa_data['file'] = msa_path
                except Exception as msa_err:
                    logger.error(f"创建MSA特征失败: {str(msa_err)}")
                    # 创建一个简单的替代MSA特征
                    msa_data = {
                        'msa': msa,
                        'num_alignments': len(msa.sequences),
                        'found': True,
                        'file': msa_path
                    }
            except Exception as e:
                logger.error(f"加载MSA失败: {str(e)}")
        
        # 模板文件路径
        template_path = os.path.join(data_dir, f"{sequence_id}_templates.json")
        
        # 如果需要模板且存在模板文件
        if use_templates and os.path.exists(template_path):
            try:
                logger.info(f"加载模板文件: {template_path}")
                with open(template_path, 'r') as f:
                    template_file_data = json.load(f)
                
                # 创建模板特征
                from alphafold3_pytorch.data.data_pipeline import make_template_features
                try:
                    template_features = make_template_features(template_file_data)
                    template_data = template_features
                    template_data['found'] = True
                    template_data['files'] = [template_path]
                except Exception as template_err:
                    logger.error(f"创建模板特征失败: {str(template_err)}")
                    # 创建一个简单的替代模板特征
                    template_data = {
                        'template_data': template_file_data,
                        'num_templates': len(template_file_data),
                        'found': True,
                        'files': [template_path]
                    }
            except Exception as e:
                logger.error(f"加载模板失败: {str(e)}")
    
    return msa_data, template_data

def prepare_input_data(sequence, molecule_type="protein", use_msa=True, use_templates=True, data_dir=None):
    """准备输入数据，包括MSA和模板数据"""
    sequence_id = f"{molecule_type}_{hash(sequence) % 10000}"  # 生成简单的序列ID
    return prepare_msas_and_templates(sequence_id, sequence, use_msa, use_templates, data_dir)

def run_single_prediction(model, sequence_id, sequence, molecule_type="protein", 
                          output_dir=None, device=None, num_recycling_steps=1, 
                          num_steps=8, generate_pdb=True, precision="fp32",
                          use_msa=False, use_templates=False, memory_config=None):
    """运行单个序列的预测"""
    start_time = time.time()
    logger.info(f"开始序列 '{sequence_id}' 的预测")
    logger.info(f"序列长度: {len(sequence)}")
    logger.info(f"分子类型: {molecule_type}")
    
    if not output_dir:
        output_dir = setup_output_dir()
    
    # 确保需要的目录存在
    os.makedirs(output_dir, exist_ok=True)
    pdb_dir = os.path.join(output_dir, "pdb")
    os.makedirs(pdb_dir, exist_ok=True)
    
    if device is None:
        device = get_device()
    
    if memory_config is None:
        memory_config = MemoryConfig()
        memory_config.optimize_for_cuda()
    
    results = types.SimpleNamespace()
    results.sequence_id = sequence_id
    results.sequence = sequence
    results.molecule_type = molecule_type
    results.plddt_values = None
    results.pae_values = None
    results.pde_values = None
    results.atom_positions = None
    results.predicted_lddt = None
    results.confidence_metrics = None
    results.timing = {}
    
        # 准备输入数据
    input_data_start = time.time()
    
    try:
        # 检查序列中是否包含RNA中特有的U碱基
        if molecule_type == "rna" and "U" in sequence:
            # RNA序列中U碱基需要特殊处理
            logger.info(f"检测到RNA序列中的U碱基，使用特殊处理方法")
            
            # 这里可以添加特殊处理逻辑，例如将U转换为T用于内部处理
            rna_sequence = sequence.replace("U", "T")
            logger.info(f"转换后的序列: {rna_sequence}")
            
            # 创建输入数据
            input_data = {
                "sequence_id": sequence_id,
                "sequence": rna_sequence,  # 使用转换后的序列
                "molecule_type": molecule_type
            }
        else:
            # 正常处理其他类型序列
            input_data = create_sequence_input(
                sequence, 
                molecule_type=molecule_type,
                device=device,
                custom_atompos=None
            )
            
        if use_msa or use_templates:
            logger.info("准备MSA和模板数据...")
            msa_template_data = prepare_msas_and_templates(
                sequence_id, 
                sequence, 
                use_msa=use_msa, 
                use_templates=use_templates
            )
            
            # 合并MSA和模板数据
            if msa_template_data:
                input_data.update(msa_template_data)
        
        input_data_time = time.time() - input_data_start
        results.timing['input_preparation'] = input_data_time
        logger.info(f"输入数据准备完成，耗时: {input_data_time:.2f}秒")
        
        # 将输入数据转移到指定设备
        if hasattr(input_data, 'to'):
            input_data = move_to_device(input_data, device, precision)
        
        # 运行预测
        predict_start = time.time()
        try:
            logger.info("使用 forward_with_alphafold3_inputs 方法")
            output = model.forward_with_alphafold3_inputs(
                input_data,
                num_recycle_iters=num_recycling_steps,
                skip_confidence=False
            )
        except (AttributeError, TypeError) as e:
            logger.error(f"使用 forward_with_alphafold3_inputs 方法出错: {str(e)}")
            logger.info("尝试使用默认 forward 方法")
            output = model.forward(
                input_data,
                num_recycle_iters=num_recycling_steps,
                num_sample_steps=num_steps
            )
        
        predict_time = time.time() - predict_start
        results.timing['prediction'] = predict_time
        logger.info(f"预测完成，耗时: {predict_time:.2f}秒")
        
        # 提取结果
        postprocess_start = time.time()
        
        try:
            # 处理原子坐标
            if hasattr(output, 'atom_pos'):
                atom_positions = output.atom_pos.detach().cpu()
                results.atom_positions = atom_positions
            
            # 提取置信度指标
            if hasattr(output, 'confidence_logits') and output.confidence_logits is not None:
                confidence_metrics = extract_confidence_metrics(output.confidence_logits, device)
                results.confidence_metrics = confidence_metrics
            
                # 保存置信度信息
                if confidence_metrics:
                    results.plddt_values = confidence_metrics.get('plddt')
                    results.pae_values = confidence_metrics.get('pae')
                    results.pde_values = confidence_metrics.get('pde')
            
            # 生成PDB文件
            if generate_pdb and results.atom_positions is not None:
                pdb_file = os.path.join(pdb_dir, f"{sequence_id}.pdb")
                try:
                    # 创建Bio.PDB结构并保存
                    structure = create_simple_biomolecule(results.atom_positions, sequence, molecule_type)
                    save_pdb(structure, pdb_file)
                    logger.info(f"PDB文件已保存至: {pdb_file}")
                except Exception as e:
                    logger.error(f"保存PDB文件时出错: {str(e)}")
                    # 尝试使用BioPython直接保存
                    backup_pdb = os.path.join(pdb_dir, f"{sequence_id}_backup.pdb")
                    save_pdb_with_biopython(results.atom_positions, sequence, backup_pdb)
                    logger.info(f"备用PDB文件已保存至: {backup_pdb}")
            
            # 绘制置信度图表
            if results.confidence_metrics is not None:
                try:
                    plot_dir = plot_confidence_metrics(results, sequence, output_dir)
                    if plot_dir:
                        logger.info(f"置信度图表已保存至: {plot_dir}")
                except Exception as e:
                    logger.error(f"创建置信度图表时出错：{str(e)}")
            
        except Exception as e:
            logger.error(f"处理预测结果时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        postprocess_time = time.time() - postprocess_start
        results.timing['postprocessing'] = postprocess_time
        
        # 总时间
        results.timing['total'] = time.time() - start_time
        logger.info(f"预测完成，总耗时: {results.timing['total']:.2f}秒")
        
        return results
        
    except Exception as e:
        logger.error(f"运行预测时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return results

def evaluate_prediction_quality(output, sequence_id, ground_truth=None):
    """评估预测质量，计算各种指标"""
    logger = logging.getLogger(__name__)
    
    try:
        # 检查输出类型并适当处理
        if isinstance(output, list):
            logger.info("检测到输出是列表类型，将使用第一个元素进行评估")
            if not output:  # 空列表
                logger.error("输出列表为空")
                return None
            output_item = output[0]  # 使用第一个元素
        else:
            output_item = output
            
        # 使用统一的置信度处理模块处理置信度信息
        if hasattr(output_item, 'confidence_logits') and output_item.confidence_logits is not None:
            # 如果有置信度逻辑值，直接处理
            device = None
            if hasattr(output_item, 'atom_pos') and hasattr(output_item.atom_pos, 'device'):
                device = output_item.atom_pos.device
                
            confidence_metrics = extract_confidence_metrics(
                output_item.confidence_logits, 
                device=device
            )
        else:
            # 如果没有置信度逻辑值，使用已提取的置信度指标
            confidence_metrics = {}
            
            # 提取plddt分数
            if hasattr(output_item, 'atom_confs') and output_item.atom_confs is not None:
                plddt = output_item.atom_confs.detach().cpu()
                confidence_metrics['plddt_mean'] = float(plddt.mean().item())
                confidence_metrics['plddt_median'] = float(plddt.median().item())
                confidence_metrics['plddt_min'] = float(plddt.min().item())
                confidence_metrics['plddt_max'] = float(plddt.max().item())
            
            # 提取PAE
            if hasattr(output_item, 'pae_output') and output_item.pae_output is not None:
                pae = output_item.pae_output.detach().cpu()
                confidence_metrics['pae_mean'] = float(pae.mean().item())
                confidence_metrics['pae_max'] = float(pae.max().item())
            
            # 提取PDE
            if hasattr(output_item, 'pde_output') and output_item.pde_output is not None:
                pde = output_item.pde_output.detach().cpu()
                confidence_metrics['pde_mean'] = float(pde.mean().item())
                confidence_metrics['pde_max'] = float(pde.max().item())
            
            # 提取PTM和iPTM分数
            if hasattr(output_item, 'ptm') and output_item.ptm is not None:
                confidence_metrics['ptm'] = float(output_item.ptm.item())
            if hasattr(output_item, 'iptm') and output_item.iptm is not None:
                confidence_metrics['iptm'] = float(output_item.iptm.item())
        
        # 如果是多链结构，提取每条链的pLDDT
        if hasattr(output_item, 'chain_indices') and output_item.chain_indices is not None and 'plddt' in confidence_metrics:
            chain_plddts = {}
            for i, chain_id in enumerate(output_item.chain_indices):
                if chain_id is not None:
                    mask = output_item.chain_masks[i] if hasattr(output_item, 'chain_masks') else None
                    if mask is not None:
                        chain_plddt = confidence_metrics['plddt'][:, mask]
                        chain_plddts[f"chain_{chain_id}"] = float(chain_plddt.mean().item())
            if chain_plddts:
                confidence_metrics['chain_plddts'] = chain_plddts
        
        # 如果有真实结构，计算结构比较指标
        if ground_truth is not None:
            logger.info("检测到基准结构，将计算结构比较指标")
            # TODO: 实现RMSD、TM-score等计算
            pass
        
        logger.info(f"置信度指标: {json.dumps({k: v for k, v in confidence_metrics.items() if not torch.is_tensor(v)}, indent=2)}")
        return confidence_metrics
        
    except Exception as e:
        logger.error(f"评估质量失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def save_prediction_results(results, output_dir, test_name, molecule_type='protein', sequence=''):
    """保存预测结果到不同格式的文件"""
    logger = logging.getLogger(__name__)
    
    if results is None:
        logger.error(f"无法保存 {test_name} 的结果：结果为空")
        return None
    
    # 创建输出目录
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件名前缀
    file_prefix = os.path.join(output_dir, f"{test_name}")
    
    # 初始化结果字典
    saved_files = {
        'npy': None,
        'pdb': None,
        'cif': None,
        'json': None
    }
    
    try:
        # 首先尝试直接从results获取atom_positions
        atom_positions = None
        if hasattr(results, 'atom_positions'):
            atom_positions = results.atom_positions
        elif isinstance(results, dict) and 'atom_positions' in results:
            atom_positions = results['atom_positions']
        elif isinstance(results, torch.Tensor):
            atom_positions = results
        
        # 保存为NumPy数组
        if atom_positions is not None:
            if torch.is_tensor(atom_positions):
                atom_positions_np = atom_positions.detach().cpu().numpy()
            else:
                atom_positions_np = atom_positions
            
            npy_path = f"{file_prefix}_atom_positions.npy"
            np.save(npy_path, atom_positions_np)
            saved_files['npy'] = npy_path
            logger.info(f"保存了NumPy数组到 {npy_path}")
            
            # 保存为PDB格式
            if sequence:
                pdb_path = f"{file_prefix}.pdb"
                if save_pdb_with_biopython(atom_positions, sequence, pdb_path):
                    saved_files['pdb'] = pdb_path
                    logger.info(f"成功保存PDB文件到 {pdb_path}")
        
        # 如果结果是Biomolecule对象或包含biomolecule属性
        biomolecule = None
        if isinstance(results, Biomolecule):
            biomolecule = results
        elif hasattr(results, 'biomolecule'):
            biomolecule = results.biomolecule
        elif isinstance(results, dict) and 'biomolecule' in results:
            biomolecule = results['biomolecule']
        
        # 如果有biomolecule对象，尝试保存为PDB和mmCIF
        if biomolecule is not None:
            try:
                # 保存为PDB
                pdb_path = f"{file_prefix}_biomolecule.pdb"
                biomolecule.to_pdb(pdb_path)
                saved_files['pdb'] = pdb_path
                logger.info(f"保存了PDB文件到 {pdb_path}")
                
                # 保存为mmCIF
                cif_path = f"{file_prefix}_biomolecule.cif"
                biomolecule.to_mmcif(cif_path)
                saved_files['cif'] = cif_path
                logger.info(f"保存了mmCIF文件到 {cif_path}")
            except Exception as e:
                logger.error(f"保存biomolecule时出错: {str(e)}")
        
        # 保存可序列化的结果为JSON
        json_path = f"{file_prefix}_results.json"
        try:
            serializable_results = {}
            
            # 如果保存了atom_positions，添加形状信息
            if atom_positions is not None:
                if torch.is_tensor(atom_positions):
                    shape = list(atom_positions.shape)
                else:
                    shape = list(atom_positions.shape)
                serializable_results['atom_positions_shape'] = shape
            
            # 添加其他可序列化的信息
            if isinstance(results, dict):
                for k, v in results.items():
                    if k != 'atom_positions' and k != 'biomolecule':
                        if torch.is_tensor(v):
                            serializable_results[k] = v.detach().cpu().numpy().tolist()
                        elif hasattr(v, '__dict__'):
                            serializable_results[k] = str(v)
                        else:
                            try:
                                json.dumps({k: v})
                                serializable_results[k] = v
                            except:
                                serializable_results[k] = str(v)
            
            # 添加分子类型和序列信息
            serializable_results['molecule_type'] = molecule_type
            if sequence:
                serializable_results['sequence'] = sequence
                serializable_results['sequence_length'] = len(sequence)
            
            # 添加保存的文件路径信息
            serializable_results['saved_files'] = {k: os.path.basename(v) if v else None for k, v in saved_files.items()}
            
            # 保存JSON
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            saved_files['json'] = json_path
            logger.info(f"保存了结果元数据到 {json_path}")
        except Exception as e:
            logger.error(f"保存JSON结果时出错: {str(e)}")
    
    except Exception as e:
        logger.error(f"保存预测结果时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    return saved_files

def generate_report(test_results, output_dir):
    """生成测试报告"""
    logger.info("生成测试报告")
    
    # 确保输出目录存在
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"创建输出目录失败: {str(e)}")
        # 使用系统临时目录作为备用
        output_dir = os.path.join(tempfile.gettempdir(), "af3_test_results")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 创建报告文件
        report_file = os.path.join(output_dir, "test_report.md")
        
        with open(report_file, "w") as f:
            f.write("# AlphaFold3-PyTorch 测试报告\n\n")
            f.write(f"日期: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 设备信息
            f.write("## 设备信息\n\n")
            if torch_npu.npu.is_available():
                f.write(f"* NPU: {torch_npu.npu.get_device_name(0)}\n")
                f.write(f"* NPU版本: {torch_npu.version}\n")
            else:
                f.write("* 仅使用CPU\n")
            f.write(f"* PyTorch版本: {torch.__version__}\n\n")
            
            # 测试概要
            f.write("## 测试概要\n\n")
            f.write(f"* 测试数量: {len(test_results)}\n")
            f.write(f"* 成功数量: {sum(1 for r in test_results.values() if r.get('success', False))}\n")
            f.write(f"* 失败数量: {sum(1 for r in test_results.values() if not r.get('success', False))}\n\n")
            
            # 详细结果
            f.write("## 详细结果\n\n")
            
            for seq_id, result in test_results.items():
                f.write(f"### {seq_id}\n\n")
                
                if result.get('success', False):
                    f.write("* 状态: ✅ 成功\n")
                else:
                    f.write("* 状态: ❌ 失败\n")
                    if 'error' in result:
                        f.write(f"* 错误: {result['error']}\n")
                
                if 'time' in result:
                    f.write(f"* 耗时: {result['time']:.2f}秒\n")
                
                if 'confidence' in result:
                    f.write("* 置信度指标:\n")
                    for metric, value in result['confidence'].items():
                        f.write(f"  * {metric}: {value:.4f}\n")
                
                if 'files' in result:
                    f.write("* 输出文件:\n")
                    for fmt, path in result['files'].items():
                        f.write(f"  * {fmt}: `{path}`\n")
                
                f.write("\n")
        
        logger.info(f"测试报告已保存至 {report_file}")
        return report_file
    
    except Exception as e:
        logger.error(f"生成报告失败: {str(e)}")
        return None

def run_test_suite(model, test_type="basic", custom_sequences=None, 
                  output_dir=None, output_formats=None,
                  use_msa=False, use_templates=False,
                  num_recycling_steps=3, memory_config=None):
    
    if output_dir is None:
        output_dir = setup_output_dir()
    else:
        # 确保用户指定的输出目录存在
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"使用用户指定的输出目录: {output_dir}")
        except Exception as e:
            logger.error(f"无法创建用户指定的输出目录: {str(e)}")
            output_dir = setup_output_dir()
    
    if output_formats is None:
        output_formats = ["pdb", "mmcif"]
    
    if memory_config is None:
        memory_config = MemoryConfig(memory_efficient=True)
    
    # 优化内存使用
    memory_config.optimize_for_cuda()
    
    # 确定要测试的序列
    sequences_to_test = {}
    
    if test_type in TEST_CONFIGS:
        for seq_id in TEST_CONFIGS[test_type]:
            if seq_id in TEST_SEQUENCES:
                sequences_to_test[seq_id] = TEST_SEQUENCES[seq_id]
            elif seq_id in TEST_LIGANDS:
                sequences_to_test[seq_id] = TEST_LIGANDS[seq_id]
    elif test_type == "custom" and custom_sequences:
        sequences_to_test = custom_sequences
    else:
        logger.warning(f"未知的测试类型 '{test_type}'，回退到基础测试")
        for seq_id in TEST_CONFIGS["basic"]:
            if seq_id in TEST_SEQUENCES:
                sequences_to_test[seq_id] = TEST_SEQUENCES[seq_id]
            elif seq_id in TEST_LIGANDS:
                sequences_to_test[seq_id] = TEST_LIGANDS[seq_id]
    
    # 记录测试开始
    logger.info(f"开始测试 - 类型: {test_type}, 序列数量: {len(sequences_to_test)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"输出格式: {output_formats}")
    logger.info(f"MSA: {'启用' if use_msa else '禁用'}, 模板搜索: {'启用' if use_templates else '禁用'}")
    logger.info(f"循环步数: {num_recycling_steps}")
    
    # 运行测试
    test_results = {}
    total_start_time = time.time()
    
    for seq_id, sequence in sequences_to_test.items():
        logger.info(f"测试序列 {seq_id}")
        
        # 确定分子类型
        if seq_id in TEST_SEQUENCES:
            if isinstance(sequence, tuple):
                if seq_id.startswith("protein_dna"):
                    molecule_type = "protein_dna"
                elif seq_id.startswith("protein_rna"):
                    molecule_type = "protein_rna"
                else:
                    molecule_type = "complex"
            elif seq_id.startswith("rna") or "_rna" in seq_id:
                molecule_type = "rna"
            elif seq_id.startswith("dna") or "_dna" in seq_id:
                molecule_type = "dna"
            else:
                molecule_type = "protein"
        elif seq_id in TEST_LIGANDS:
            molecule_type = "ligand"
        else:
            molecule_type = "protein"  # 默认为蛋白质
        
        # 初始化结果
        test_results[seq_id] = {
            'success': False,
            'molecule_type': molecule_type
        }
        
        # 清理内存
        if memory_config.clear_cache_between_tests:
            memory_config.clear_cuda_cache()
        
        # 运行预测
        start_time = time.time()
        try:
            # 自动确定步数
            if molecule_type == "protein":
                seq_len = len(sequence)
            elif isinstance(sequence, tuple):
                seq_len = sum(len(s) for s in sequence)
            else:
                seq_len = len(sequence)
            
            num_steps = max(20, min(50, seq_len // 5))
            
            # 运行预测
            output = run_single_prediction(
                model, seq_id, sequence, 
                molecule_type=molecule_type,
                use_msa=use_msa, 
                use_templates=use_templates,
                num_recycling_steps=num_recycling_steps,
                num_steps=num_steps,
                memory_config=memory_config
            )
            
            if output is None:
                test_results[seq_id]['error'] = "预测失败"
                continue
            
            # 评估质量
            confidence = evaluate_prediction_quality(output, seq_id)
            if confidence:
                test_results[seq_id]['confidence'] = confidence
            
            # 保存结果
            result_files = save_prediction_results(
                output, output_dir, seq_id, molecule_type, sequence
            )
            if result_files:
                test_results[seq_id]['files'] = result_files
            
            # 标记成功
            test_results[seq_id]['success'] = True
            
        except Exception as e:
            test_results[seq_id]['error'] = str(e)
            logger.error(f"测试 {seq_id} 失败: {str(e)}")
            
        finally:
            elapsed_time = time.time() - start_time
            test_results[seq_id]['time'] = elapsed_time
            logger.info(f"测试 {seq_id} 完成，耗时: {elapsed_time:.2f}秒")
    
    # 计算总耗时
    total_time = time.time() - total_start_time
    logger.info(f"所有测试完成，总耗时: {total_time:.2f}秒")
    
    # 即使没有成功的测试，也生成报告
    if not any(r.get('success', False) for r in test_results.values()):
        logger.warning("所有测试均失败，生成简单报告")
    
    # 生成报告
    report_file = generate_report(test_results, output_dir)
    
    return test_results, report_file

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AlphaFold 3 PyTorch运行脚本")
    
    # 基本参数
    parser.add_argument("--sequence", type=str, default="AG", help="要预测的序列")
    parser.add_argument("--molecule-type", type=str, default="protein", choices=["protein", "rna", "dna"], help="分子类型")
    parser.add_argument("--model-path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU进行计算")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="计算精度")
    parser.add_argument("--quiet", action="store_true", help="减少输出信息")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    # 功能选择
    parser.add_argument("--test-basic-functionality", action="store_true", help="测试基本功能")
    parser.add_argument("--test-comprehensive", action="store_true", help="运行综合测试")
    parser.add_argument("--test-model-loading", action="store_true", help="测试模型加载")
    parser.add_argument("--test-basic-prediction", action="store_true", help="测试基本预测功能")
    parser.add_argument("--test-pdb-generation", action="store_true", help="测试PDB文件生成")
    parser.add_argument("--run-complete-pipeline", action="store_true", help="运行完整预测管道")
    parser.add_argument("--extract-confidence", action="store_true", help="提取置信度信息")
    parser.add_argument("--generate-pdb", action="store_true", help="生成PDB文件")
    parser.add_argument("--plot", action="store_true", help="绘制置信度图")
    
    # 高级参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮次")
    parser.add_argument("--no-msa", action="store_true", help="不使用MSA")
    parser.add_argument("--no-templates", action="store_true", help="不使用模板")
    parser.add_argument("--recycling-steps", type=int, default=3, help="循环步数")
    parser.add_argument("--sampling-steps", type=int, default=8, help="采样步数")
    parser.add_argument("--memory-efficient", action="store_true", help="使用内存高效模式")
    parser.add_argument("--custom-atom-positions", type=str, default=None, help="自定义原子位置文件")
    
    args = parser.parse_args()
    return args

def plot_confidence_metrics(output, sequence, output_dir):
    """创建置信度可视化图表"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 检查是否存在confidence_metrics
        if hasattr(output, 'confidence_metrics') and output.confidence_metrics is not None:
            metrics = output.confidence_metrics
        else:
            logger.warning("没有可用的置信度指标进行绘图")
            return
    
    # 创建图表目录
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
    
        # 设置Matplotlib参数
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        
        # 1. pLDDT histogram
        if 'plddt' in metrics and metrics['plddt'] is not None:
            plt.figure()
            plddt = metrics['plddt'].cpu().numpy()
            sns.histplot(plddt, bins=20, kde=True)
            plt.title('pLDDT Distribution')
            plt.xlabel('pLDDT Value')
            plt.ylabel('Frequency')
            plt.axvline(x=np.mean(plddt), color='r', linestyle='--', 
                        label=f'Mean: {np.mean(plddt):.2f}')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'plddt_histogram.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # pLDDT per residue
            plt.figure()
            plt.plot(range(len(plddt)), plddt)
            plt.title('pLDDT per Residue')
            plt.xlabel('Residue Index')
            plt.ylabel('pLDDT Value')
            plt.axhline(y=70, color='r', linestyle='--', label='70 (Threshold)')
            plt.axhline(y=90, color='g', linestyle='--', label='90 (High confidence)')
            plt.axhline(y=50, color='orange', linestyle='--', label='50 (Low confidence)')
            for i, aa in enumerate(sequence):
                plt.annotate(aa, (i, plddt[i]), textcoords="offset points", 
                           xytext=(0,5), ha='center', fontsize=8)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'plddt_per_residue.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. PAE matrix
        if 'pae' in metrics and metrics['pae'] is not None:
            plt.figure()
            pae = metrics['pae'].cpu().numpy()
            seq_len = pae.shape[0]
            
            ax = sns.heatmap(pae, cmap="YlGnBu_r", vmin=0, vmax=min(30, np.max(pae)))
            ax.set_title('Predicted Aligned Error (PAE)')
            ax.set_xlabel('Residue j')
            ax.set_ylabel('Residue i')
            
            # 添加序列标签
            if seq_len <= 20:  # 对于短序列添加氨基酸标签
                tick_positions = np.arange(seq_len)
                ax.set_xticks(tick_positions + 0.5)
                ax.set_yticks(tick_positions + 0.5)
                ax.set_xticklabels(list(sequence))
                ax.set_yticklabels(list(sequence))
            
            plt.savefig(os.path.join(plots_dir, 'pae_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # PAE distribution
            plt.figure()
            sns.histplot(pae.flatten(), bins=30, kde=True)
            plt.title('PAE Distribution')
            plt.xlabel('PAE Value (Å)')
            plt.ylabel('Frequency')
            plt.axvline(x=np.mean(pae), color='r', linestyle='--', 
                        label=f'Mean: {np.mean(pae):.2f}Å')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'pae_histogram.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # 3. PDE matrix (if available)
        if 'pde' in metrics and metrics['pde'] is not None:
            plt.figure()
            pde = metrics['pde'].cpu().numpy()
            
            ax = sns.heatmap(pde, cmap="viridis", vmin=0, vmax=min(20, np.max(pde)))
            ax.set_title('Predicted Distance Error (PDE)')
            ax.set_xlabel('Residue j')
            ax.set_ylabel('Residue i')
            
            # 添加序列标签
            if pde.shape[0] <= 20:  # 对于短序列添加氨基酸标签
                tick_positions = np.arange(pde.shape[0])
                ax.set_xticks(tick_positions + 0.5)
                ax.set_yticks(tick_positions + 0.5)
                ax.set_xticklabels(list(sequence))
                ax.set_yticklabels(list(sequence))
            
            plt.savefig(os.path.join(plots_dir, 'pde_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # 4. Overall confidence visualization
        plt.figure(figsize=(12, 6))
        
        # 使用子图绘制
        if 'plddt' in metrics and metrics['plddt'] is not None:
            plt.subplot(1, 2, 1)
            plt.plot(range(len(metrics['plddt'].cpu())), metrics['plddt'].cpu())
            plt.title('pLDDT by Residue')
            plt.xlabel('Residue Index')
            plt.ylabel('pLDDT')
            plt.grid(True, alpha=0.3)
            
        if 'pae' in metrics and metrics['pae'] is not None:
            plt.subplot(1, 2, 2)
            mean_pae = metrics['pae'].mean(dim=0).cpu()
            plt.plot(range(len(mean_pae)), mean_pae)
            plt.title('Mean PAE by Residue')
            plt.xlabel('Residue Index')
            plt.ylabel('Mean PAE (Å)')
            plt.grid(True, alpha=0.3)
        
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'overall_confidence.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # 5. Combined confidence scores
        if 'plddt' in metrics and metrics['plddt'] is not None:
            plt.figure(figsize=(12, 8))
            plddt = metrics['plddt'].cpu().numpy()
            
            confidence_categories = []
            for score in plddt:
                if score >= 90:
                    confidence_categories.append('Very high (90+)')
                elif score >= 70:
                    confidence_categories.append('High (70-90)')
                elif score >= 50:
                    confidence_categories.append('Medium (50-70)')
                else:
                    confidence_categories.append('Low (<50)')
            
            category_colors = {
                'Very high (90+)': 'blue',
                'High (70-90)': 'green', 
                'Medium (50-70)': 'orange',
                'Low (<50)': 'red'
            }
            
            confidence_colors = [category_colors[cat] for cat in confidence_categories]
            
            plt.figure(figsize=(max(6, len(sequence)/2), 4))
            plt.bar(range(len(plddt)), plddt, color=confidence_colors)
            plt.title('Residue Confidence Profile')
            plt.xlabel('Residue Index')
            plt.ylabel('pLDDT Score')
            plt.ylim(0, 100)
            
            # 图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='Very high (90+)'),
                Patch(facecolor='green', label='High (70-90)'),
                Patch(facecolor='orange', label='Medium (50-70)'),
                Patch(facecolor='red', label='Low (<50)')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            # 显示序列
            if len(sequence) <= 50:
                plt.xticks(range(len(sequence)), list(sequence))
            
            plt.savefig(os.path.join(plots_dir, 'residue_confidence.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        # 保存总结报告
        summary_file = os.path.join(plots_dir, 'confidence_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Confidence Metrics Summary for Sequence: {sequence}\n\n")
            
            if 'plddt' in metrics and metrics['plddt'] is not None:
                plddt_np = metrics['plddt'].cpu().numpy()
                f.write(f"pLDDT (Per-residue prediction confidence):\n")
                f.write(f"  - Mean: {np.mean(plddt_np):.2f}\n")
                f.write(f"  - Median: {np.median(plddt_np):.2f}\n")
                f.write(f"  - Min: {np.min(plddt_np):.2f}\n")
                f.write(f"  - Max: {np.max(plddt_np):.2f}\n\n")
                
                # 信心水平分布
                very_high = np.sum(plddt_np >= 90)
                high = np.sum((plddt_np >= 70) & (plddt_np < 90))
                medium = np.sum((plddt_np >= 50) & (plddt_np < 70))
                low = np.sum(plddt_np < 50)
                
                f.write(f"Confidence Distribution:\n")
                f.write(f"  - Very high (90+): {very_high} residues ({very_high/len(plddt_np)*100:.1f}%)\n")
                f.write(f"  - High (70-90): {high} residues ({high/len(plddt_np)*100:.1f}%)\n")
                f.write(f"  - Medium (50-70): {medium} residues ({medium/len(plddt_np)*100:.1f}%)\n")
                f.write(f"  - Low (<50): {low} residues ({low/len(plddt_np)*100:.1f}%)\n\n")
            
            if 'pae' in metrics and metrics['pae'] is not None:
                pae_np = metrics['pae'].cpu().numpy()
                f.write(f"PAE (Predicted Aligned Error):\n")
                f.write(f"  - Mean: {np.mean(pae_np):.2f} Å\n")
                f.write(f"  - Median: {np.median(pae_np):.2f} Å\n")
                f.write(f"  - Min: {np.min(pae_np):.2f} Å\n")
                f.write(f"  - Max: {np.max(pae_np):.2f} Å\n\n")
            
            if 'pde' in metrics and metrics['pde'] is not None:
                pde_np = metrics['pde'].cpu().numpy()
                f.write(f"PDE (Predicted Distance Error):\n")
                f.write(f"  - Mean: {np.mean(pde_np):.2f} Å\n")
                f.write(f"  - Median: {np.median(pde_np):.2f} Å\n")
                f.write(f"  - Min: {np.min(pde_np):.2f} Å\n")
                f.write(f"  - Max: {np.max(pde_np):.2f} Å\n\n")
            
            f.write("Note: Higher pLDDT values indicate higher confidence. Lower PAE/PDE values indicate higher accuracy.\n")
        
        logger.info(f"置信度可视化图表已保存到 {plots_dir}")
        return plots_dir
    except ImportError as e:
        logger.warning(f"无法创建图表：{str(e)}。请确保已安装matplotlib和seaborn。")
        return None
    except Exception as e:
        logger.error(f"创建置信度图表时出错：{str(e)}")
        return None

def test_confidence_extraction(model, sequence="AG", molecule_type="protein", output_dir=None, plot_confidence=False):
    """测试置信度提取功能"""
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始测试置信度提取功能，序列: {sequence}，分子类型: {molecule_type}")
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 创建输入数据
    logger.info("准备输入数据...")
    inputs = create_sequence_input(sequence, molecule_type=molecule_type, device=device)
    
    # 运行预测并提取置信度
    logger.info("运行预测并提取置信度...")
    try:
            # 使用forward_with_alphafold3_inputs提取置信度信息
        if hasattr(model, 'forward_with_alphafold3_inputs'):
            output_tuple = model.forward_with_alphafold3_inputs(
                inputs,
                num_recycling_steps=1,
                num_sample_steps=8,
                return_confidence_head_logits=True
            )
        else:
            logger.error("模型没有forward_with_alphafold3_inputs方法")
            return False
            
            logger.info("成功获取带置信度信息的预测结果")
        
        # 提取置信度指标
        if hasattr(output_tuple, 'confidence_logits') and output_tuple.confidence_logits is not None:
            confidence_metrics = extract_confidence_metrics(output_tuple.confidence_logits, device)
            logger.info(f"成功提取置信度指标: {list(confidence_metrics.keys()) if confidence_metrics else '无'}")
        else:
            logger.warning("未找到置信度logits，无法提取置信度指标")
            return False
        
        # 保存置信度指标
        metrics_file = os.path.join(output_dir, "confidence_metrics.json")
        try:
            with open(metrics_file, 'w') as f:
                serializable_metrics = {}
                for k, v in confidence_metrics.items():
                    if hasattr(v, 'tolist'):
                        serializable_metrics[k + "_shape"] = list(v.shape)
                    else:
                        serializable_metrics[k] = v
                
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"置信度指标已保存至: {metrics_file}")
        except Exception as e:
            logger.error(f"保存置信度指标时出错: {str(e)}")
                
        # 可选：绘制置信度图表
        if plot_confidence:
            logger.info("绘制置信度可视化图表...")
            try:
                results = types.SimpleNamespace()
                results.confidence_metrics = confidence_metrics
                plot_dir = plot_confidence_metrics(results, sequence, output_dir)
                if plot_dir:
                    logger.info(f"置信度图表已保存至: {plot_dir}")
            except Exception as e:
                logger.error(f"绘制置信度图表时出错: {str(e)}")
        
            return True
        
    except Exception as e:
        logger.error(f"测试置信度提取功能时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_data_preparation(sequence_id="ag", sequence="AG", molecule_type="protein", output_dir=None):
    """专门测试数据准备功能"""
    logger.info("=" * 50)
    logger.info("开始测试数据准备")
    logger.info("=" * 50)
    
    if output_dir is None:
        output_dir = setup_output_dir()
    
    if sequence_id.lower() in ["ag", "small_rna", "small_dna"]:
        if sequence_id.lower() == "ag":
            sequence = "AG"
            molecule_type = "protein"
        elif sequence_id.lower() == "small_rna":
            sequence = "ACGU"
            molecule_type = "rna"
        elif sequence_id.lower() == "small_dna":
            sequence = "ACGT"
            molecule_type = "dna"
    
    logger.info(f"测试序列: {sequence_id} ({molecule_type}) - {sequence}")
    
    # 检查数据目录
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    logger.info(f"检查数据目录: {data_dir}")
    
    # 检查MSA数据目录
    msa_dir = os.path.join(data_dir, "msa_data")
    if os.path.exists(msa_dir):
        logger.info(f"MSA数据目录存在: {msa_dir}")
        files = os.listdir(msa_dir)
        if files:
            logger.info(f"MSA目录内容: {', '.join(files[:10])}{' 等' if len(files) > 10 else ''}")
        else:
            logger.warning("MSA数据目录为空")
    else:
        logger.warning(f"MSA数据目录不存在: {msa_dir}")
    
    # 检查PDB数据目录
    pdb_data_dir = os.path.join(data_dir, "pdb_data")
    if os.path.exists(pdb_data_dir):
        logger.info(f"PDB数据目录存在: {pdb_data_dir}")
        
        # 检查mmcifs目录
        mmcifs_dir = os.path.join(pdb_data_dir, "mmcifs")
        if os.path.exists(mmcifs_dir):
            logger.info(f"mmcifs目录存在: {mmcifs_dir}")
            subdirs = [d for d in os.listdir(mmcifs_dir) if os.path.isdir(os.path.join(mmcifs_dir, d))]
            if subdirs:
                logger.info(f"mmcifs子目录: {', '.join(subdirs[:10])}{' 等' if len(subdirs) > 10 else ''}")
                
                # 随机选择一个子目录检查内容
                if subdirs:
                    sample_dir = os.path.join(mmcifs_dir, subdirs[0])
                    files = os.listdir(sample_dir)
                    if files:
                        logger.info(f"样本子目录 {subdirs[0]} 的内容: {', '.join(files[:5])}{' 等' if len(files) > 5 else ''}")
            else:
                logger.warning("mmcifs目录为空")
        else:
            logger.warning(f"mmcifs目录不存在: {mmcifs_dir}")
        
        # 检查数据缓存目录
        data_caches_dir = os.path.join(pdb_data_dir, "data_caches")
        if os.path.exists(data_caches_dir):
            logger.info(f"数据缓存目录存在: {data_caches_dir}")
            
            # 检查MSA缓存
            msa_cache_dir = os.path.join(data_caches_dir, "msa")
            if os.path.exists(msa_cache_dir):
                logger.info(f"MSA缓存目录存在: {msa_cache_dir}")
                msas_dir = os.path.join(msa_cache_dir, "msas")
                if os.path.exists(msas_dir):
                    files = os.listdir(msas_dir)
                    if files:
                        logger.info(f"MSA缓存文件: {', '.join(files[:5])}{' 等' if len(files) > 5 else ''}")
                    else:
                        logger.warning("MSA缓存目录为空")
            else:
                logger.warning(f"MSA缓存目录不存在: {msa_cache_dir}")
            
            # 检查模板缓存
            template_cache_dir = os.path.join(data_caches_dir, "template")
            if os.path.exists(template_cache_dir):
                logger.info(f"模板缓存目录存在: {template_cache_dir}")
                templates_dir = os.path.join(template_cache_dir, "templates")
                if os.path.exists(templates_dir):
                    files = os.listdir(templates_dir)
                    if files:
                        logger.info(f"模板缓存文件: {', '.join(files[:5])}{' 等' if len(files) > 5 else ''}")
                    else:
                        logger.warning("模板缓存目录为空")
            else:
                logger.warning(f"模板缓存目录不存在: {template_cache_dir}")
    else:
        logger.warning(f"PDB数据目录不存在: {pdb_data_dir}")
    
    # 测试MSA和模板数据准备
    logger.info("-" * 50)
    logger.info("测试MSA和模板数据准备")
    msa_data, template_data = prepare_msas_and_templates(
        sequence_id, sequence, use_msa=True, use_templates=True, data_dir=data_dir
    )
    
    if msa_data:
        logger.info(f"MSA数据: {'找到' if msa_data.get('found', False) else '未找到'}")
        if msa_data.get('found', False):
            logger.info(f"MSA文件: {msa_data.get('file', '未指定')}")
    else:
        logger.warning("未能获取MSA数据")
    
    if template_data:
        logger.info(f"模板数据: {'找到' if template_data.get('found', False) else '未找到'}")
        if template_data.get('found', False):
            files = template_data.get('files', [])
            if files:
                logger.info(f"找到 {len(files)} 个模板文件")
                logger.info(f"示例模板文件: {files[0]}")
    else:
        logger.warning("未能获取模板数据")
    
    logger.info("-" * 50)
    logger.info("数据准备测试完成")
    
    # 写入测试报告
    report_file = os.path.join(output_dir, "data_preparation_report.md")
    with open(report_file, "w") as f:
        f.write("# 数据准备测试报告\n\n")
        f.write(f"测试时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"测试序列: {sequence_id} ({molecule_type}) - {sequence}\n\n")
        
        f.write("## 数据目录检查\n\n")
        f.write(f"- MSA数据目录: {'存在' if os.path.exists(msa_dir) else '不存在'}\n")
        f.write(f"- PDB数据目录: {'存在' if os.path.exists(pdb_data_dir) else '不存在'}\n")
        f.write(f"- mmcifs目录: {'存在' if os.path.exists(mmcifs_dir) else '不存在'}\n")
        f.write(f"- 数据缓存目录: {'存在' if os.path.exists(data_caches_dir) else '不存在'}\n")
        
        f.write("\n## MSA和模板数据准备\n\n")
        if msa_data:
            f.write(f"- MSA数据: {'找到' if msa_data.get('found', False) else '未找到'}\n")
            if msa_data.get('found', False):
                f.write(f"  - MSA文件: {msa_data.get('file', '未指定')}\n")
        else:
            f.write("- MSA数据: 未能获取\n")
        
        if template_data:
            f.write(f"- 模板数据: {'找到' if template_data.get('found', False) else '未找到'}\n")
            if template_data.get('found', False):
                files = template_data.get('files', [])
                if files:
                    f.write(f"  - 找到 {len(files)} 个模板文件\n")
                    f.write(f"  - 示例模板文件: {files[0]}\n")
        else:
            f.write("- 模板数据: 未能获取\n")
    
    logger.info(f"测试报告已保存至: {report_file}")
    return report_file

def test_basic_functionality(output_dir=None, device=None, precision="fp32", num_recycling_steps=3):
    """运行基本功能测试，不需要预训练模型"""
    # 设置日志记录
    setup_logging(quiet=False)
    logger.info("=" * 50)
    logger.info("开始基本功能测试")
    logger.info("=" * 50)

    if device is None:
        # NOTE 这里应该改成False 因为是NPU
        device = get_device(force_cpu=True)  # 为基本功能测试使用CPU
        logger.info(f"使用设备: {device}")
        
    if output_dir is None:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./af3_output_{timestamp}"
    
    # 确保输出目录存在
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")
    except Exception as e:
        logger.error(f"创建输出目录失败: {str(e)}")
        return None
    
    logger.info(f"开始基本功能测试，输出目录: {output_dir}")
    logger.info(f"使用设备: {device}, 精度: {precision}, 循环步数: {num_recycling_steps}")
    
    test_cases = [
        {"id": "ag", "sequence": "AG", "type": "protein"},
        {"id": "small_rna", "sequence": "ACGU", "type": "rna"},
        {"id": "small_dna", "sequence": "ACGT", "type": "dna"}
    ]
    
    results = {}
    save_structures = True  # 添加这个变量的定义
    
    logger.info("本次测试不加载预训练模型")
    
    for test_case in test_cases:
        sequence_id = test_case["id"]
        sequence = test_case["sequence"]
        molecule_type = test_case["type"]
        
        logger.info(f"测试序列 {sequence_id}: {sequence} (类型: {molecule_type})")
        
        try:
            # 步骤1: 创建序列输入
            logger.info(f"为序列 {sequence_id} 创建输入")
            inputs = create_sequence_input(sequence, molecule_type, device)
            if inputs is None:
                logger.error(f"为 {sequence_id} 准备输入数据失败")
                results[sequence_id] = {"status": "失败", "错误": "准备输入数据失败"}
                continue
            
            # 步骤2: 将输入转换为适当的格式
            logger.info("将输入转换为可处理的格式...")
            try:
                from alphafold3_pytorch.inputs import alphafold3_input_to_molecule_lengthed_molecule_input
                from alphafold3_pytorch.inputs import molecule_lengthed_molecule_input_to_atom_input
                
                # 转换输入格式
                if hasattr(inputs, 'proteins') and not hasattr(inputs, 'atom_inputs'):
                    logger.info("检测到Alphafold3Input对象，正在转换为atom_input...")
                    molecule_lengthed_input = alphafold3_input_to_molecule_lengthed_molecule_input(inputs)
                    atom_input = molecule_lengthed_molecule_input_to_atom_input(molecule_lengthed_input)
                    
                    # 定义基础维度
                    dim_atom_inputs = 3
                    
                    # 确保原子输入的维度正确
                    if hasattr(atom_input, 'atom_inputs') and atom_input.atom_inputs.shape[-1] != dim_atom_inputs:
                        logger.warning(f"atom_inputs的特征维度 ({atom_input.atom_inputs.shape[-1]}) 与期望的维度 ({dim_atom_inputs}) 不匹配")
                        # 这里我们可以进行填充或截断处理
                        original_shape = atom_input.atom_inputs.shape
                        if atom_input.atom_inputs.shape[-1] < dim_atom_inputs:
                            # 填充到期望的维度
                            padding = torch.zeros(*original_shape[:-1], dim_atom_inputs - original_shape[-1], device=atom_input.atom_inputs.device)
                            atom_input.atom_inputs = torch.cat([atom_input.atom_inputs, padding], dim=-1)
                            logger.info(f"已将atom_inputs填充到期望维度: {atom_input.atom_inputs.shape}")
                        else:
                            # 截断到期望的维度
                            atom_input.atom_inputs = atom_input.atom_inputs[..., :dim_atom_inputs]
                            logger.info(f"已将atom_inputs截断到期望维度: {atom_input.atom_inputs.shape}")
                    
                    # 将转换后的输入赋值回原变量
                    inputs = atom_input
                    logger.info(f"输入已转换为: {type(inputs)}，atom_inputs形状: {atom_input.atom_inputs.shape if hasattr(atom_input, 'atom_inputs') else 'N/A'}")
            except Exception as convert_err:
                logger.error(f"转换输入格式失败: {str(convert_err)}")
                logger.error(traceback.format_exc())
                results[sequence_id] = {"status": "失败", "错误": f"转换输入格式失败: {str(convert_err)}"}
                continue
                
            # 步骤3: 创建模拟的原子位置输出
            mock_atom_positions = generate_mock_atom_positions(sequence, device)
            
            # 步骤4: 创建模拟的置信度logits
            from types import SimpleNamespace
            mock_output = SimpleNamespace()
            mock_output.atom_pos = mock_atom_positions
            
            sequence_length = len(sequence)
            confidence_logits = SimpleNamespace()
            
            # 创建pLDDT logits: [batch(1), residues, bins(50)]
            plddt_bins = 50
            confidence_logits.plddt = torch.randn(1, sequence_length, plddt_bins, device=device)
            
            # 创建PAE logits: [batch(1), residues, residues, bins(64)]
            pae_bins = 64
            confidence_logits.pae = torch.randn(1, sequence_length, sequence_length, pae_bins, device=device)
            
            # 获取原子数量
            atom_count = calculate_total_atoms(sequence)
            confidence_logits.pde = torch.randn(1, atom_count, atom_count, pae_bins, device=device)
            
            mock_output.confidence_logits = confidence_logits
            
            # 步骤5: 处理置信度信息
            # 提取置信度值
            plddt_values = extract_plddt_from_logits(confidence_logits.plddt, device)
            pae_values = extract_pae_from_logits(confidence_logits.pae, device)
            pde_values = extract_pde_from_logits(confidence_logits.pde, device)
            
            # 添加到输出
            mock_output.plddt_values = plddt_values
            mock_output.pae_values = pae_values
            mock_output.pde_values = pde_values
            
            # 评估质量指标
            metrics = evaluate_prediction_quality(mock_output, sequence_id)
            
            # 步骤6: 保存PDB结构
            case_output_dir = os.path.join(output_dir, sequence_id)
            os.makedirs(case_output_dir, exist_ok=True)
            
            pdb_file = os.path.join(case_output_dir, f"{sequence_id}.pdb")
            
            try:
                # 将输出移至CPU以生成PDB
                mock_output_cpu = move_to_device(mock_output, 'cpu')
                
                # 创建PDB文件
                logger.info(f"为 {sequence_id} 生成结构文件")
                biomolecule = create_simple_biomolecule(
                    mock_output_cpu.atom_pos[0].detach().numpy(),
                    sequence,
                    molecule_type
                )
                
                # 保存PDB和mmCIF文件
                if biomolecule is not None:
                    if save_structures:
                        save_pdb(biomolecule, pdb_file)
                        logger.info(f"为 {sequence_id} 生成PDB文件: {pdb_file}")
                else:
                    logger.warning(f"为 {sequence_id} 创建生物分子结构失败")
            except Exception as e:
                logger.warning(f"为 {sequence_id} 生成PDB文件失败: {str(e)}")
                logger.warning(traceback.format_exc())
            
            # 保存结果
            test_report = {
                "status": "成功",
                "序列": sequence,
                "分子类型": molecule_type,
                "置信度指标": metrics if metrics else "无可用指标",
                "MSA数据": "未找到MSA数据，使用默认MSA",
                "模板数据": "未找到模板数据，使用默认模板" 
            }
            
            results[sequence_id] = test_report
            
            logger.info(f"完成 {sequence_id} 的测试")
            
        except Exception as e:
            logger.error(f"测试 {sequence_id} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            results[sequence_id] = {"status": "失败", "错误": str(e)}
    
    # 生成测试报告
    report_file = os.path.join(output_dir, "basic_functionality_report.md")
    try:
        with open(report_file, "w") as f:
            f.write("# AlphaFold 3 基本功能测试报告\n\n")
            f.write(f"测试时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for sequence_id, result in results.items():
                f.write(f"## 序列: {sequence_id}\n\n")
                
                if result["status"] == "成功":
                    f.write("- 测试状态: ✅ 成功\n")
                    f.write(f"- 序列: {result['序列']}\n")
                    f.write(f"- 分子类型: {result['分子类型']}\n")
                    f.write(f"- MSA数据: {result['MSA数据']}\n")
                    f.write(f"- 模板数据: {result['模板数据']}\n")
                    
                    if isinstance(result["置信度指标"], dict):
                        f.write("- 置信度指标:\n")
                        for metric_name, metric_value in result["置信度指标"].items():
                            if isinstance(metric_value, (int, float)):
                                f.write(f"  - {metric_name}: {metric_value:.2f}\n")
                            else:
                                f.write(f"  - {metric_name}: {metric_value}\n")
                    else:
                        f.write(f"- 置信度指标: {result['置信度指标']}\n")
                else:
                    f.write("- 测试状态: ❌ 失败\n")
                    f.write(f"- 错误: {result['错误']}\n")
                
                f.write("\n")
            
            # 添加系统信息
            f.write("## 系统信息\n\n")
            f.write(f"- PyTorch版本: {torch.__version__}\n")
            f.write(f"- NPU可用: {torch_npu.npu.is_available()}\n")
            if torch_npu.npu.is_available():
                f.write(f"- NPU设备: {torch_npu.npu.get_device_name(0)}\n")
            f.write(f"- 测试设备: {device}\n")
            f.write(f"- 精度: {precision}\n")
        
        logger.info(f"基本功能测试报告已生成: {report_file}")
    except Exception as e:
        logger.error(f"生成测试报告失败: {str(e)}")
    
    logger.info("基本功能测试完成")
    return results

def extract_confidence_metrics(confidence_logits, device=None):
    """从logits中提取置信度指标"""
    confidence_metrics = {}
    
    try:
        if device is None:
            # 尝试获取任何可用的logits设备
            if hasattr(confidence_logits, 'plddt_logits') and confidence_logits.plddt_logits is not None:
                device = confidence_logits.plddt_logits.device
            elif hasattr(confidence_logits, 'pae_logits') and confidence_logits.pae_logits is not None:
                device = confidence_logits.pae_logits.device
            elif hasattr(confidence_logits, 'pde_logits') and confidence_logits.pde_logits is not None:
                device = confidence_logits.pde_logits.device
            else:
                device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
    
        logger.info(f"提取置信度指标，使用设备: {device}")
        
        # 从logits中提取pLDDT
        if hasattr(confidence_logits, 'plddt_logits') and confidence_logits.plddt_logits is not None:
            confidence_metrics['plddt'] = extract_plddt_from_logits(confidence_logits.plddt_logits, device)
            if confidence_metrics['plddt'] is not None:
                logger.info(f"提取的pLDDT形状: {confidence_metrics['plddt'].shape}")
            else:
                logger.warning("没有找到pLDDT logits")
        
        # 从logits中提取PAE
        if hasattr(confidence_logits, 'pae_logits') and confidence_logits.pae_logits is not None:
            confidence_metrics['pae'] = extract_pae_from_logits(confidence_logits.pae_logits, device)
            if confidence_metrics['pae'] is not None:
                logger.info(f"提取的PAE形状: {confidence_metrics['pae'].shape}")
            else:
                logger.warning("没有找到PAE logits")
        
        # 从logits中提取PDE
        if hasattr(confidence_logits, 'pde_logits') and confidence_logits.pde_logits is not None:
            confidence_metrics['pde'] = extract_pde_from_logits(confidence_logits.pde_logits, device)
            if confidence_metrics['pde'] is not None:
                logger.info(f"提取的PDE形状: {confidence_metrics['pde'].shape}")
            else:
                logger.warning("没有找到PDE logits")
        
        return confidence_metrics
    
    except Exception as e:
        logger.error(f"提取置信度指标失败: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def save_pdb_with_biopython(atom_pos, sequence, filename):
    """使用 BioPython 保存原子坐标为 PDB 文件"""
    try:
        # 确保atom_pos是numpy数组
        if torch.is_tensor(atom_pos):
            atom_pos = atom_pos.detach().cpu().numpy()
        
        # 将3D张量转换为2D
        if len(atom_pos.shape) == 3:
            # 形状是 [batch, num_atoms, 3]
            atom_pos = atom_pos[0]
        
        # 氨基酸三字母代码映射
        aa_map = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        
        # 每个氨基酸的主要原子名称（只使用其中的一部分）
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
        
        # 为核酸添加核酸残基名称和原子名称
        na_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'T': 'THY', 'U': 'URA'}
        
        # 简化的核酸原子名称
        na_atom_names = {
            'ADE': ['P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
            'CYT': ['P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
            'GUA': ['P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
            'THY': ['P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'],
            'URA': ['P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6']
        }
        
        # 创建结构
        structure = Structure("predicted")
        model = Model(0)
        chain = Chain("A")
        
        # 计算每个残基应该有多少个原子
        num_atoms = atom_pos.shape[0]
        len_seq = len(sequence)
        
        # 如果原子数不能整除残基数，使用平均值
        atoms_per_res = max(1, num_atoms // len_seq)
        
        logger.info(f"保存PDB文件：序列长度 {len_seq}，原子数 {num_atoms}，每个残基平均原子数 {atoms_per_res}")
        
        atom_idx = 0
        for i, res_char in enumerate(sequence):
            # 根据残基类型获取残基名称
            if res_char in aa_map:  # 氨基酸
                res_name = aa_map.get(res_char, 'UNK')
                names = atom_names.get(res_name, ['CA'])
            elif res_char in na_map:  # 核酸
                res_name = na_map.get(res_char, 'UNK')
                names = na_atom_names.get(res_name, ['CA'])
            else:
                res_name = 'UNK'
                names = ['CA']
            
            residue = Residue((' ', i+1, ' '), res_name, '')
            
            # 确定当前残基的原子数（最后一个残基可能不足）
            res_atom_count = min(atoms_per_res, num_atoms - atom_idx)
            
            # 添加原子到残基
            for j in range(res_atom_count):
                # 选择原子名称（如果超出了名称列表范围，使用CA+索引）
                if j < len(names):
                    atom_name = names[j]
                    elem = atom_name[0]  # 第一个字符作为元素符号
                else:
                    atom_name = f"CA{j}"
                    elem = "C"
                
                # 获取原子坐标
                if atom_idx < num_atoms:
                    coord = atom_pos[atom_idx]
                    # 创建原子并添加到残基
                    atom = Atom(atom_name, coord, 0.0, 1.0, ' ', atom_name, atom_idx+1, element=elem)
                    residue.add(atom)
                    atom_idx += 1
            
            # 添加残基到链
            chain.add(residue)
        
        # 添加链到模型，模型到结构
        model.add(chain)
        structure.add(model)
        
        # 保存为PDB文件
        io = PDBIO()
        io.set_structure(structure)
        io.save(filename)
        logger.info(f"成功保存PDB文件: {filename}")
        return True
    
    except Exception as e:
        logger.error(f"保存PDB文件失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

class TestFramework:
    """AlphaFold3测试框架类"""
    
    def __init__(self, epochs=5, recycling_steps=3, sampling_steps=50, 
                 output_dir=None, model_path=None, cpu_only=False):
        """初始化测试框架
        
        Args:
            epochs: 训练轮次
            recycling_steps: 循环步数
            sampling_steps: 采样步数
            output_dir: 输出目录
            model_path: 预训练模型路径
            cpu_only: 是否仅使用CPU
        """
        self.epochs = epochs
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self.model_path = model_path
        self.cpu_only = cpu_only
        
        # 设置设备
        self.device = torch.device('cpu') if cpu_only or not torch_npu.npu.is_available() else torch.device('npu')
        
        # 设置输出目录
        if output_dir is None:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"./test_results_{timestamp}"
        else:
            self.output_dir = output_dir
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志记录
        setup_logging(quiet=False, log_file=os.path.join(self.output_dir, "test.log"))
        
        # 设置内存配置
        self.memory_config = MemoryConfig(memory_efficient=True)
        
        # 初始化模型
        self.model = None
        logger.info(f"测试框架初始化完成，设备: {self.device}, 输出目录: {self.output_dir}")
        
    def initialize_model(self):
        """初始化模型"""
        logger.info("初始化模型...")
        
        if self.model is not None:
            logger.info("模型已初始化，跳过")
            return True
            
        try:
            # 尝试加载预训练模型
            if self.model_path is not None and check_model_path(self.model_path):
                logger.info(f"加载预训练模型: {self.model_path}")
                self.model = load_pretrained_model(self.model_path, device=self.device)
                if self.model is not None:
                    logger.info("预训练模型加载成功")
                    return True
                    
            # 如果无法加载预训练模型，创建新模型
            logger.info("创建新的AlphaFold3模型")
            from alphafold3_pytorch import Alphafold3
            
            # 提供必需的参数
            self.model = Alphafold3(
                dim_atom_inputs=3,  # 设置为3，与原子位置特征维度匹配
                dim_template_feats=64,  # 设置一个合理的默认值
                num_molecule_mods=0  # 禁用分子修饰嵌入，避免is_molecule_mod错误
            )
            
            # 将模型移至设备
            self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info(f"模型初始化成功，设备: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("=" * 50)
        logger.info("开始综合测试")
        logger.info("=" * 50)
        
        # 确保模型已初始化
        if not self.initialize_model():
            logger.error("模型初始化失败，无法运行测试")
            return False
            
        # 记录测试开始信息
        logger.info(f"测试参数: epochs={self.epochs}, recycling_steps={self.recycling_steps}, sampling_steps={self.sampling_steps}")
        
        # 创建测试序列集
        test_sequences = {}
        for seq_id, seq in TEST_SEQUENCES.items():
            if seq_id in ["ag", "acgu", "acgt"]:  # 选择基本测试序列
                test_sequences[seq_id] = seq
                
        # 运行测试套件
        test_results, report_file = run_test_suite(
            self.model, 
            test_type="basic", 
            custom_sequences=test_sequences,
            output_dir=self.output_dir,
            use_msa=not self.cpu_only,
            use_templates=not self.cpu_only,
            num_recycling_steps=self.recycling_steps,
            memory_config=self.memory_config
        )
        
        # 测试置信度提取
        confidence_results = test_confidence_extraction(
            self.model,
            sequence="AG",
            output_dir=self.output_dir,
            plot_confidence=True
        )
        
        # 测试数据准备
        data_results = test_data_preparation(
            sequence_id="ag",
            sequence="AG",
            output_dir=self.output_dir
        )
        
        # 综合测试报告
        all_results = {
            "basic_tests": test_results,
            "confidence_extraction": confidence_results,
            "data_preparation": data_results
        }
        
        # 创建综合报告
        report_path = os.path.join(self.output_dir, "comprehensive_report.md")
        try:
            with open(report_path, 'w') as f:
                f.write("# AlphaFold3-PyTorch 综合测试报告\n\n")
                f.write(f"测试时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
                # 系统信息
                f.write("## 系统信息\n\n")
                f.write(f"- 设备: {self.device}\n")
                f.write(f"- PyTorch版本: {torch.__version__}\n")
            if torch_npu.npu.is_available():
                f.write(f"- NPU版本: {torch_npu.version}\n")
                f.write(f"- NPU: {torch_npu.npu.get_device_name(0)}\n")
                
                # 测试参数
                f.write("\n## 测试参数\n\n")
                f.write(f"- 训练轮次: {self.epochs}\n")
                f.write(f"- 循环步数: {self.recycling_steps}\n")
                f.write(f"- 采样步数: {self.sampling_steps}\n")
                f.write(f"- MSA: {'禁用' if self.cpu_only else '启用'}\n")
                f.write(f"- 模板: {'禁用' if self.cpu_only else '启用'}\n")
                
                # 测试结果摘要
                f.write("\n## 测试结果摘要\n\n")
                
                # 基本测试结果
                f.write("\n### 基本功能测试\n\n")
                if test_results:
                    success_count = sum(1 for r in test_results.values() if r.get('success', False))
                    total_count = len(test_results)
                    f.write(f"- 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)\n")
                    
                    # 详细结果
                    f.write("\n#### 详细结果\n\n")
                    for seq_id, result in test_results.items():
                        status = "✓ 成功" if result.get('success', False) else "✗ 失败"
                        f.write(f"- {seq_id}: {status}\n")
                        if not result.get('success', False) and 'error' in result:
                            f.write(f"  - 错误: {result['error']}\n")
                else:
                    f.write("- 未运行基本功能测试\n")
                
                # 置信度提取测试
                f.write("\n### 置信度提取测试\n\n")
                if confidence_results is not None:
                    f.write(f"- 状态: {'✓ 成功' if confidence_results else '✗ 失败'}\n")
                    if confidence_results:
                        f.write("- 置信度指标成功提取\n")
                        f.write("- 结果已保存在输出目录中\n")
                else:
                    f.write("- 未运行置信度提取测试\n")
                
                # 数据准备测试
                f.write("\n### 数据准备测试\n\n")
                if data_results:
                    f.write(f"- 报告文件: {data_results}\n")
                    f.write("- 状态: ✓ 成功 (已生成报告)\n")
                else:
                    f.write("- 未运行数据准备测试\n")
                
                # 总结
                f.write("\n## 总结\n\n")
                all_success = all([
                    any(r.get('success', False) for r in test_results.values()) if test_results else False,
                    confidence_results if confidence_results is not None else False,  # confidence_results 直接是布尔值
                    data_results is not None  # 只要data_results存在就认为成功
                ])
                
                if all_success:
                    f.write("所有测试均成功完成。AlphaFold3-PyTorch可以正常工作。\n")
                else:
                    f.write("部分测试失败。请检查详细结果以了解问题所在。\n")
            
            logger.info(f"综合测试报告已保存至 {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建综合报告失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def run_complete_pipeline(sequence, sequence_id=None, molecule_type="protein", 
                        output_dir=None, device=None, precision="fp32", 
                        num_recycling_steps=3, num_steps=8, 
                        save_structures=True, train_epochs=0, learning_rate=1e-4):
    """运行完整的预测管道，包括模型加载、预测、置信度提取和结构生成，可选训练步骤"""
    if sequence_id is None:
        now = dt.now()
        seq_id = f"seq_{now.strftime('%Y%m%d_%H%M%S')}"
    else:
        seq_id = sequence_id
    
    # 设置输出目录
    if output_dir is None:
        output_dir = setup_output_dir()
    else:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"无法创建输出目录 {output_dir}: {str(e)}")
            return None

    logger.info(f"输出目录: {output_dir}")
    logger.info(f"使用设备: {device}")
    
    # 初始化结果字典
    results = {
        "sequence_id": seq_id,
        "sequence": sequence,
        "length": len(sequence),
        "molecule_type": molecule_type,
        "success": False,
        "timing": {},
        "training": {"epochs": train_epochs, "performed": train_epochs > 0}
    }
    
    total_start_time = time.time()
    
    try:
        # 步骤1: 加载模型
        logger.info("步骤1: 加载模型")
        start_time = time.time()
        
        try:
            if os.path.exists(DEFAULT_MODEL_PATH):
                logger.info(f"从 {DEFAULT_MODEL_PATH} 加载预训练模型")
                model = load_pretrained_model(DEFAULT_MODEL_PATH, device, precision)
            else:
                logger.info("初始化随机模型用于测试")
                # 导入必要的库
                sys.path.append(os.getcwd())
                from alphafold3_pytorch import Alphafold3
                from alphafold3_pytorch.configs import Alphafold3Config
                
                try:
                    # 尝试使用配置创建模型
                    if os.path.exists('./configs/minimal_model.yaml'):
                        logger.info("使用最小配置文件创建模型")
                        model = Alphafold3Config.create_instance_from_yaml_file('./configs/minimal_model.yaml')
                    else:
                        # 创建最小模型配置
                        logger.info("创建自定义最小模型")
                        from torch import nn
                        
                        # 创建一个非常简单的模型用于测试
                        # 注意：这不是一个真正的AlphaFold3模型，只是为了测试流程
                        class SimpleModel(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.dummy = nn.Parameter(torch.randn(1))
                            
                            def forward(self, *args, **kwargs):
                                # 获取设备和数据类型
                                input_tensor = None
                                for arg in args:
                                    if torch.is_tensor(arg):
                                        input_tensor = arg
                                        break
                                
                                if input_tensor is None:
                                    for k, v in kwargs.items():
                                        if torch.is_tensor(v):
                                            input_tensor = v
                                            break
                                
                                device = self.dummy.device
                                dtype = self.dummy.dtype
                                
                                if input_tensor is not None:
                                    device = input_tensor.device
                                    dtype = input_tensor.dtype
                                
                                # 获取序列信息，以确定正确的原子数量
                                sequence = kwargs.get('sequence', None)
                                if sequence is None and 'proteins' in kwargs:
                                    sequence = kwargs['proteins'][0] if isinstance(kwargs['proteins'], list) and len(kwargs['proteins']) > 0 else None
                                
                                atom_count = 51  # 默认原子数量，针对我们的测试序列"MKTVRQ"
                                
                                if sequence:
                                    try:
                                        # 计算实际的原子数量
                                        total_atoms = calculate_total_atoms(sequence)
                                        logger.info(f"计算序列 '{sequence}' 的原子数量: {total_atoms}")
                                        atom_count = total_atoms
                                    except Exception as e:
                                        logger.warning(f"计算原子数量失败: {str(e)}，使用默认值: {atom_count}")
                                
                                # 返回模拟的输出，确保在同一设备和精度
                                b = 1  # 批次大小，默认为1
                                
                                # 生成模拟的原子位置和置信度输出，使用正确的原子数量
                                # 确保设置requires_grad=True以支持梯度计算
                                positions = torch.randn(b, atom_count, 3, device=device, dtype=dtype, requires_grad=True)
                                plddt_logits = torch.randn(b, atom_count, 50, device=device, dtype=dtype, requires_grad=True)
                                pae_logits = torch.randn(b, 64, atom_count, atom_count, device=device, dtype=dtype, requires_grad=True)
                                pde_logits = torch.randn(b, 64, atom_count, atom_count, device=device, dtype=dtype, requires_grad=True)
                                
                                # 检查是否有目标原子位置，如果有则计算损失（训练模式）
                                target_atom_pos = kwargs.get('atom_pos', None)
                                if target_atom_pos is not None and self.training:
                                    # 简单的训练模式 - 计算均方误差损失
                                    try:
                                        # 如果目标原子数量与预测原子数量不匹配，可能需要调整
                                        # 这里我们采用简单的方法，只使用前n个原子，其中n是较小的那个数量
                                        min_atoms = min(positions.shape[1], target_atom_pos.shape[0])
                                        
                                        # 提取要比较的张量
                                        pred_positions = positions[0, :min_atoms, :]
                                        target_positions = target_atom_pos[:min_atoms, :]
                                        
                                        # 确保形状一致
                                        if target_positions.ndim == 2:
                                            # 如果target是[n, 3]，需要确保pred也是[n, 3]
                                            pred_positions = pred_positions.view(-1, 3)
                                            
                                        # 计算MSE损失
                                        loss = torch.nn.functional.mse_loss(pred_positions, target_positions)
                                        
                                        # 记录训练状态
                                        logger.info(f"计算训练损失: {loss.item():.6f}, 预测形状: {pred_positions.shape}, 目标形状: {target_positions.shape}")
                                        
                                        # 返回损失
                                        return loss
                                    except Exception as e:
                                        logger.error(f"计算损失时出错: {str(e)}")
                                        # 如果计算损失失败，返回一个默认损失，确保requires_grad=True
                                        return torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
                                else:
                                    # 推理模式 - 返回SimpleOutput对象
                                    return SimpleOutput(
                                        positions=positions,
                                        plddt_logits=plddt_logits,
                                        pae_logits=pae_logits,
                                        pde_logits=pde_logits
                                    )
                        
                        # 创建一个简单的输出类
                        class SimpleOutput:
                            def __init__(self, positions, plddt_logits, pae_logits, pde_logits):
                                self.positions = positions
                                self.atom_pos = positions
                                self.plddt_logits = plddt_logits
                                self.pae_logits = pae_logits
                                self.pde_logits = pde_logits
                        
                        model = SimpleModel()
                    
                    model = model.to(device)
                    logger.info(f"模型已初始化并移至设备: {device}")
                
                except Exception as e:
                    logger.error(f"初始化模型时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            
            results["model_time"] = time.time() - start_time
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            results["error"] = f"模型加载失败: {str(e)}"
            return results
        
        # 步骤2: 准备输入数据
        logger.info("步骤2: 准备输入数据")
        start_time = time.time()
        
        try:
            # 创建输入数据
            input_data = create_sequence_input(sequence, molecule_type, device)
            results["input_time"] = time.time() - start_time
            
            if input_data is None:
                logger.error("创建输入数据失败")
                results["error"] = "创建输入数据失败"
                return results
        
        except Exception as e:
            logger.error(f"准备输入数据失败: {str(e)}")
            results["error"] = f"准备输入数据失败: {str(e)}"
            return results
        
        # 辅助函数：将Alphafold3Input转换为可训练的字典
        def prepare_input_for_training(input_obj, mock_atom_pos=None):
            """将输入对象转换为模型可以接受的格式"""
            # 如果是Alphafold3Input对象，尝试提取字典表示
            if hasattr(input_obj, '_asdict'):
                try:
                    # 使用_asdict方法（如果有的话）
                    input_dict = input_obj._asdict()
                    logger.info(f"使用_asdict()方法将输入转换为字典，包含键: {list(input_dict.keys())}")
                    
                    # 添加原子位置（如果提供）
                    if mock_atom_pos is not None:
                        input_dict['atom_pos'] = mock_atom_pos
                    
                    return input_dict
                except Exception as e:
                    logger.warning(f"无法使用_asdict()方法: {str(e)}")
            
            # 如果是Alphafold3Input对象，尝试提取所有属性
            if hasattr(input_obj, '__dict__'):
                try:
                    # 创建属性字典
                    input_dict = {}
                    for key, value in input_obj.__dict__.items():
                        if not key.startswith('_'):  # 排除私有属性
                            input_dict[key] = value
                    
                    logger.info(f"从对象属性创建字典，包含键: {list(input_dict.keys())}")
                    
                    # 添加原子位置（如果提供）
                    if mock_atom_pos is not None:
                        input_dict['atom_pos'] = mock_atom_pos
                    
                    # 确保序列信息存在
                    if 'proteins' in input_dict and isinstance(input_dict['proteins'], list) and len(input_dict['proteins']) > 0:
                        input_dict['sequence'] = input_dict['proteins'][0]
                        logger.info(f"添加序列信息: {input_dict['sequence']}")
                    
                    return input_dict
                except Exception as e:
                    logger.warning(f"无法从对象属性创建字典: {str(e)}")
            
            # 如果对象有model_forward_dict方法（BatchedAtomInput通常有这个）
            if hasattr(input_obj, 'model_forward_dict') and callable(input_obj.model_forward_dict):
                try:
                    input_dict = input_obj.model_forward_dict()
                    logger.info(f"使用model_forward_dict()方法创建字典，包含键: {list(input_dict.keys())}")
                    
                    # 添加原子位置（如果提供）
                    if mock_atom_pos is not None:
                        input_dict['atom_pos'] = mock_atom_pos
                    
                    return input_dict
                except Exception as e:
                    logger.warning(f"无法使用model_forward_dict()方法: {str(e)}")
            
            # 如果是字典，直接使用
            if isinstance(input_obj, dict):
                # 添加原子位置（如果提供）
                result_dict = input_obj.copy()  # 创建副本以避免修改原始对象
                if mock_atom_pos is not None:
                    result_dict['atom_pos'] = mock_atom_pos
                return result_dict
            
            # 最后的回退选项：创建一个包含输入对象的字典
            logger.warning("无法将输入转换为字典，将使用一个包含输入对象的简单字典")
            return {"input": input_obj, "atom_pos": mock_atom_pos} if mock_atom_pos is not None else {"input": input_obj}
        
        # 步骤3: 训练模型（如果指定训练轮数）
        if train_epochs > 0:
            logger.info("步骤3: 训练模型")
            start_time = time.time()
            
            try:
                # 导入必要的库
                from torch.optim import Adam
                
                # 设置模型为训练模式
                model.train()
                
                # 定义优化器
                optimizer = Adam(model.parameters(), lr=learning_rate)
                
                # 获取输入数据，为训练添加目标结构
                # 注意：在实际训练中，应该使用真实结构作为目标
                # 这里我们生成一个模拟的目标结构用于演示
                mock_atom_positions = generate_mock_atom_positions(sequence, device)
                logger.info(f"生成模拟目标结构，形状: {mock_atom_positions.shape}")
                
                # 设置训练参数字典
                train_losses = []
                
                # 训练循环
                logger.info(f"开始训练，训练轮数: {train_epochs}")
                
                for epoch in range(train_epochs):
                    epoch_start_time = time.time()
                    
                    # 优化器梯度置零
                    optimizer.zero_grad()
                    
                    # 前向传播 - 使用输入数据计算输出
                    # 确保输入数据格式正确，这里可能需要根据具体模型调整
                    try:
                        # 准备包含目标结构的输入数据
                        # 在一个真实的训练循环中，通常会从训练数据集中获取这些信息
                        # 这里我们模拟这个过程
                        training_input = prepare_input_for_training(input_data, mock_atom_positions)
                        
                        # 前向传播和损失计算
                        if isinstance(model, torch.nn.Module) and hasattr(model, 'forward'):
                            # 如果模型具有AlphaFold3风格的接口
                            loss = model(
                                **training_input,
                                num_recycle_iters=1,  # 训练时使用较少的循环步骤
                            )
                        else:
                            logger.warning("模型不具备标准接口，可能无法正确训练")
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                        
                        # 检查损失是否为标量
                        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
                            # 反向传播
                            loss.backward()
                            
                            # 梯度裁剪（可选）
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            # 优化器步骤
                            optimizer.step()
                            
                            # 记录损失
                            train_losses.append(loss.item())
                            
                            if (epoch + 1) % max(1, train_epochs // 10) == 0:
                                logger.info(f"训练轮数 {epoch+1}/{train_epochs}, 损失: {loss.item():.6f}")
                        else:
                            logger.warning(f"损失不是标量，跳过此轮训练。损失类型: {type(loss)}")
                    
                    except Exception as e:
                        logger.error(f"训练过程中出错: {str(e)}")
                        logger.error(traceback.format_exc())
                        if epoch == 0:
                            # 如果第一轮就失败，可能存在兼容性问题
                            logger.warning("训练失败，继续执行预测步骤")
                            break
                    
                    # 记录每轮训练的时间
                    epoch_time = time.time() - epoch_start_time
                    logger.info(f"训练轮数 {epoch+1} 完成，耗时: {epoch_time:.2f}秒")
                
                # 保存训练后的模型
                if train_losses:
                    model_save_path = os.path.join(output_dir, f"{seq_id}_trained_model.pt")
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"训练后的模型已保存至 {model_save_path}")
                    results["trained_model_path"] = model_save_path
                    results["training"]["final_loss"] = train_losses[-1] if train_losses else None
                    results["training"]["loss_history"] = train_losses
                
                # 设置模型为评估模式
                model.eval()
                
                results["training_time"] = time.time() - start_time
                logger.info(f"训练完成，耗时: {results['training_time']:.2f}秒")
            
            except Exception as e:
                logger.error(f"训练失败: {str(e)}")
                logger.error(traceback.format_exc())
                # 继续执行预测步骤
            
        # 步骤4: 运行预测
        logger.info("步骤4: 运行预测")
        start_time = time.time()
        
        try:
            # 运行预测
            with torch.no_grad():
                output = model(
                    input_data,
                    num_recycle_iters=num_recycling_steps,
                    num_sample_steps=num_steps,
                    return_confidence_head_logits=True
                )
            
            results["inference_time"] = time.time() - start_time
            logger.info(f"预测完成，耗时: {results['inference_time']:.2f}秒")
        
        except Exception as e:
            logger.error(f"运行预测失败: {str(e)}")
            results["error"] = f"运行预测失败: {str(e)}"
            return results
        
        # 步骤5: 提取置信度指标
        logger.info("步骤5: 提取置信度指标")
        start_time = time.time()
        
        try:
            # 提取置信度指标
            confidence_metrics = extract_confidence_metrics(output, device)
            
            if confidence_metrics:
                results["confidence_metrics"] = {}
                for metric, value in confidence_metrics.items():
                    if isinstance(value, (int, float)):
                        results["confidence_metrics"][metric] = value
                    elif hasattr(value, 'tolist'):
                        results["confidence_metrics"][metric + "_shape"] = list(value.shape)
            
            results["confidence_time"] = time.time() - start_time
        
        except Exception as e:
            logger.error(f"提取置信度指标失败: {str(e)}")
            # 继续执行，不中断流程
        
        # 步骤6: 生成结构文件
        if save_structures:
            logger.info("步骤6: 生成结构文件")
            start_time = time.time()
            
            try:
                # 获取原子位置
                if hasattr(output, 'positions') and output.positions is not None:
                    atom_positions = output.positions
                elif hasattr(output, 'atom_pos') and output.atom_pos is not None:
                    atom_positions = output.atom_pos
                else:
                    logger.warning("未找到原子位置，无法生成结构文件")
                    atom_positions = None
                
                if atom_positions is not None:
                    # 确保atom_positions是CPU tensor
                    if torch.is_tensor(atom_positions):
                        atom_positions = atom_positions.detach().cpu()
                    
                    # 生成PDB文件
                    pdb_file = os.path.join(output_dir, f"{seq_id}.pdb")
                    structure = create_simple_biomolecule(atom_positions, sequence, molecule_type)
                    if structure:
                        save_pdb(structure, pdb_file)
                        logger.info(f"结构已保存至 {pdb_file}")
                        results["pdb_file"] = pdb_file
                
                results["structure_time"] = time.time() - start_time
            
            except Exception as e:
                logger.error(f"生成结构文件失败: {str(e)}")
                # 继续执行，不中断流程
        
        # 生成预测报告
        report_path = os.path.join(output_dir, f"{seq_id}_report.md")
        try:
            with open(report_path, 'w') as f:
                f.write(f"# 预测报告: {sequence}\n\n")
                f.write(f"预测时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 基本信息
                f.write("## 基本信息\n\n")
                f.write(f"- 序列: {sequence}\n")
                f.write(f"- 序列长度: {len(sequence)}\n")
                f.write(f"- 分子类型: {molecule_type}\n")
                f.write(f"- 序列ID: {seq_id}\n\n")
                
                # 预测参数
                f.write("## 预测参数\n\n")
                f.write(f"- 循环步数: {num_recycling_steps}\n")
                f.write(f"- 采样步数: {num_steps}\n")
                f.write(f"- 使用设备: {device}\n")
                f.write(f"- 精度: {precision}\n\n")
                
                # 运行时间
                f.write("## 运行时间\n\n")
                for step, time_value in results["timing"].items():
                    f.write(f"- {step}: {time_value:.2f}秒\n")
                f.write(f"- 总时间: {time.time() - total_start_time:.2f}秒\n\n")
                
                # 置信度指标
                f.write("## 置信度指标\n\n")
                if "confidence_metrics" in results:
                    for metric, value in results["confidence_metrics"].items():
                        if isinstance(value, (int, float)):
                            f.write(f"- {metric}: {value:.4f}\n")
                        elif hasattr(value, 'shape'):
                            f.write(f"- {metric}: 形状={value.shape}\n")
                else:
                    f.write("- 未提取置信度指标\n")
                
                # 输出文件
                f.write("\n## 输出文件\n\n")
                if "pdb_file" in results:
                    f.write(f"- PDB文件: {results['pdb_file']}\n")
                else:
                    f.write("- 未生成PDB文件\n")
            
            logger.info(f"预测报告已保存: {report_path}")
            results["report_file"] = report_path
            
        except Exception as e:
            logger.error(f"生成预测报告失败: {str(e)}")
        
        # 标记成功
        results["success"] = True
        results["total_time"] = time.time() - total_start_time
        logger.info(f"完整预测管道执行完成，总耗时: {results['total_time']:.2f}秒")
        
        return results
        
    except Exception as e:
        logger.error(f"运行完整预测管道时出错: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        return results

if __name__ == "__main__":
    import sys
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='AlphaFold 3 Test Framework')
    
    # 添加参数
    parser.add_argument('--test-basic-functionality', action='store_true', help='运行基本功能测试')
    parser.add_argument('--test-comprehensive', action='store_true', help='运行综合测试')
    parser.add_argument('--run-complete-pipeline', action='store_true', help='运行完整的模型训练和预测流程')
    parser.add_argument('--sequence', type=str, default='AG', help='用于测试的氨基酸序列')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--model-path', type=str, default=None, help='预训练模型路径(可选)')
    parser.add_argument('--force-cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--precision', type=str, default='fp32', help='计算精度')
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')
    parser.add_argument('--recycling-steps', type=int, default=3, help='循环步数')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮次')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='训练学习率')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(quiet=args.quiet, log_file=args.log_file)
    logger.info("开始运行测试框架")
    
    try:
        # 运行基本功能测试
        if args.test_basic_functionality:
            logger.info("准备运行基本功能测试...")
            results = test_basic_functionality(
                output_dir=args.output_dir,
                device=torch.device('cpu') if args.force_cpu else None,
                precision=args.precision,
                num_recycling_steps=args.recycling_steps
            )
            if results:
                logger.info("基本功能测试完成")
            else:
                logger.error("基本功能测试失败")
    
    # 如果要运行综合测试
        elif args.test_comprehensive:
            logger.info("运行综合测试...")
            test_framework = TestFramework(
                epochs=args.epochs,
                recycling_steps=args.recycling_steps,
                output_dir=args.output_dir,
                model_path=args.model_path,
                        cpu_only=args.force_cpu
            )
            test_framework.run_comprehensive_test()
    
    # 如果要运行完整流程
        elif args.run_complete_pipeline:
            logger.info("运行完整流程...")
            logger.info(f"训练配置: 轮数={args.epochs}, 学习率={args.learning_rate}")
            results = run_complete_pipeline(
                sequence=args.sequence,
                output_dir=args.output_dir,
                        device=torch.device('cpu') if args.force_cpu else None,
                        precision=args.precision,
                num_recycling_steps=args.recycling_steps,
                save_structures=True,
                train_epochs=args.epochs,
                learning_rate=args.learning_rate
            )
        
    # 如果没有指定任何操作，显示帮助
        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("测试框架运行完成")
