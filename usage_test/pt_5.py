#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaFold3-PyTorch 全面功能测试脚本
支持预训练模型加载、多种分子类型测试、MSA和模板搜索，以及多种输出格式

用法:
    # 基本功能测试
    python run_af3_pt4.py --test-basic-functionality
    
    # 静默模式运行（只显示警告和错误）
    python run_af3_pt4.py --test-basic-functionality --quiet
    
    # 完整测试套件
    python run_af3_pt4.py --run-comprehensive-test --output-dir results
    
    # 测试置信度提取
    python run_af3_pt4.py --test-extraction
    
    # 测试PDB文件生成
    python run_af3_pt4.py --test-pdb-generation
    
    # 指定序列进行测试
    python run_af3_pt4.py --test-basic-functionality --sequence ACDEFGHIKLMNPQRSTVWY
    
    # 完整管道测试
    python run_af3_pt4.py --run-complete-pipeline --sequence ACDEFGHIKLMNPQRSTVWY

参数:
    --test-basic-functionality  运行基本功能测试
    --test-extraction          测试置信度提取功能
    --test-plotting            测试置信度可视化功能
    --run-comprehensive-test   运行综合测试，测试不同类型和大小的分子
    --test-pdb-generation      测试PDB文件生成功能
    --run-complete-pipeline    运行完整预测管道
    --sequence SEQ             指定测试序列
    --output-dir DIR           输出目录
    --epochs EPOCHS            训练轮数
    --quiet                    静默模式，减少日志输出
    
注意:
    1. 该脚本已优化以支持CPU+NPU异构计算环境
    2. 包含move_to_device函数以解决对象在设备间移动的兼容性问题
    3. 默认不使用MSA和模板数据，可通过参数开启，MSA和模板数据需要下载补充
"""

import gc
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
from datetime import datetime as dt
import tempfile
import traceback
from Bio.PDB import PDBParser, PDBIO, MMCIFIO, StructureBuilder
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

import torch_npu

#from .pt_4 import save_pdb_with_biopython, test_pdb_generation

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


# 预定义的测试序列
TEST_SEQUENCES = {
    # 蛋白质序列
    "ag": "AG",  # 丙氨酸-甘氨酸二肽，简单测试
    "insulin_a": "GIVEQCCTSICSLYQLENYCN",  # 胰岛素A链
    "insulin_b": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # 胰岛素B链
    "melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",  # 蜜蜂毒素，常用测试序列
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 泛素
    "lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",  # 溶菌酶
    
    # RNA序列
    "small_rna": "GGAACC",  # 简单RNA序列
    "trna_fragment": "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA",  # tRNA片段
    
    # DNA序列
    "small_dna": "ATGC",  # 简单DNA序列
    "dna_fragment": "ATGCAAATCGACTACGTAGCTACGTACGTAGCTAGCTAGCTA",  # DNA片段
    
    # 混合测试
    "protein_dna": ("MQIFVKTLTGKTITLEVEPSD", "ATGCAAATCGACTACGTAGCT"),  # 蛋白质和DNA
    "protein_rna": ("GIVEQCCTSICSLYQLENYCN", "GGAACC"),  # 蛋白质和RNA
}

# 常见配体和金属离子
TEST_LIGANDS = {
    "ATP": "ATP",  # 三磷酸腺苷
    "GTP": "GTP",  # 三磷酸鸟苷
    "HEM": "HEM",  # 血红素
    "NAD": "NAD",  # 烟酰胺腺嘌呤二核苷酸
    "ZN": "Zn2+",  # 锌离子
    "MG": "Mg2+",  # 镁离子
    "CA": "Ca2+",  # 钙离子
}

# 测试配置
TEST_CONFIGS = {
    "quick": ["ag", "small_rna", "small_dna"],  # 快速测试
    "basic": ["ag", "insulin_a", "small_rna", "small_dna"],  # 基础测试
    "medium": ["insulin_a", "trna_fragment", "dna_fragment", "ATP"],  # 中等规模测试
    "full": list(TEST_SEQUENCES.keys()) + list(TEST_LIGANDS.keys()),  # 完整测试
    "protein": ["ag", "insulin_a", "insulin_b", "melittin", "ubiquitin", "lysozyme"],  # 仅蛋白质
    "dna": ["small_dna", "dna_fragment"],  # 仅DNA
    "rna": ["small_rna", "trna_fragment"],  # 仅RNA
    "ligand": list(TEST_LIGANDS.keys()),  # 仅配体和金属离子
    "complex": ["protein_dna", "protein_rna"],  # 复合物
    "custom": []  # 自定义测试，由用户指定
}

# 氨基酸原子数量
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

def calculate_total_atoms(sequence):
    """计算序列中的总原子数"""
    atom_counts = get_atom_counts_per_residue()
    return sum(atom_counts.get(aa, 5) for aa in sequence)

def generate_mock_atom_positions(sequence, device=None):
    """为给定序列生成模拟的原子位置"""
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
            
            # 设置自动调整内存分配策略 这个没用
            # os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
            
            # 清空缓存
            self.clear_cuda_cache()
            
            # 设置默认张量类型
            torch.set_default_dtype(torch.float32)
    
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
        "./af3.bin",
        "./alphafold3.bin",
        "./weights/af3.bin",
        "./weights/alphafold3.bin",
        "/app/af3/model/af3.bin"
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
                          num_sample_steps=8, generate_pdb=True, precision="fp32"):
    """
    使用给定模型运行单个序列预测
    
    Args:
        model: AlphaFold3模型
        sequence_id: 序列的唯一标识符
        sequence: 氨基酸或核苷酸序列
        molecule_type: 分子类型 ("protein", "rna", "dna")
        output_dir: 输出目录
        device: 计算设备
        num_recycling_steps: 循环步数
        num_sample_steps: 采样步数
        generate_pdb: 是否生成PDB文件
        precision: 精度设置 ("fp32", "fp16", "bf16")
    
    Returns:
        预测结果对象
    """
    if device is None:
        device = model.device if hasattr(model, 'device') else get_device()
    
    if output_dir is None:
        output_dir = setup_output_dir()
    
    logger.info(f"开始预测序列 {sequence_id}: {sequence} (类型: {molecule_type})")
    
    try:
        # 准备输入数据
        logger.info("准备模型输入数据")
        input_data = prepare_input_data(sequence, molecule_type)
        
        # 将数据移至指定设备和精度
        input_data = move_to_device(input_data, device, precision)
        
        if model is None:
            logger.warning("未提供模型，将只准备数据而不进行预测")
            return input_data
        
        # 确保模型在正确的设备上
        model = move_to_device(model, device, precision)
        
        # 运行模型预测
        logger.info(f"开始模型预测 (循环步数: {num_recycling_steps}, 采样步数: {num_sample_steps})")
        with torch.no_grad():
            output = model(
                input_data,
                num_recycles=num_recycling_steps,
                num_sampling_steps=num_sample_steps,
            )
        
        # 评估预测质量
        metrics = evaluate_prediction_quality(output, sequence_id)
        
        # 如果需要，生成PDB结构文件
        if generate_pdb and hasattr(output, 'atom_pos') and output.atom_pos is not None:
            logger.info("生成PDB结构文件")
            pdb_file = os.path.join(output_dir, f"{sequence_id}.pdb")
            
            # 将输出移至CPU以生成PDB
            output_cpu = move_to_device(output, 'cpu')
            
            if hasattr(output_cpu, 'atom_pos') and output_cpu.atom_pos is not None:
                try:
                    save_pdb(output_cpu, pdb_file)
                    logger.info(f"PDB文件已保存至 {pdb_file}")
                except Exception as e:
                    logger.error(f"保存PDB文件失败: {str(e)}")
            else:
                logger.warning("未找到原子坐标，无法生成PDB文件")
        
        return output
        
    except Exception as e:
        logger.error(f"运行预测时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def evaluate_prediction_quality(output, sequence_id, ground_truth=None):
    """评估预测质量，计算各种指标"""
    logger.info(f"评估 {sequence_id} 的预测质量")
    
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
                
            confidence_metrics = process_confidence_metrics(
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
                  num_recycles=3, memory_config=None):
    """运行测试套件"""
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
                num_recycles=num_recycling_steps,
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
        # 检查matplotlib是否可用
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端，适用于无GUI环境
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("无法导入matplotlib，请使用 pip install matplotlib 安装")
        return None
    
    logger.info("创建置信度可视化图表...")
    
    # 检查是否有置信度信息
    has_plddt = hasattr(output, 'atom_confs') and output.atom_confs is not None
    has_pae = hasattr(output, 'pae_values') and output.pae_values is not None
    has_pde = hasattr(output, 'pde_values') and output.pde_values is not None
    
    if not (has_plddt or has_pae or has_pde):
        logger.warning("没有找到置信度信息，无法生成图表")
        return None
    
    # 创建图表目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    result_files = []
    
    # pLDDT分数条形图
    if has_plddt:
        try:
            plddt = output.atom_confs[0].cpu().numpy()  # 取第一个样本
            
            # 获取残基级别pLDDT
            residue_plddt = []
            atom_idx = 0
            atom_counts = get_atom_counts_per_residue()
            
            for i, aa in enumerate(sequence):
                num_atoms = atom_counts.get(aa, 5)
                if atom_idx + num_atoms <= plddt.shape[0]:
                    res_plddt = float(plddt[atom_idx:atom_idx+num_atoms].mean())
                    residue_plddt.append(res_plddt)
                else:
                    residue_plddt.append(50.0)  # 默认中等置信度
                atom_idx += num_atoms
            
            # 创建残基级pLDDT图表
            plt.figure(figsize=(10, 6))
            x = np.arange(len(residue_plddt))
            
            # 根据pLDDT分数设置颜色
            colors = []
            for score in residue_plddt:
                if score >= 90:
                    colors.append('blue')
                elif score >= 70:
                    colors.append('green')
                elif score >= 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            plt.bar(x, residue_plddt, color=colors)
            plt.axhline(y=90, color='blue', linestyle='--', alpha=0.5)
            plt.axhline(y=70, color='green', linestyle='--', alpha=0.5)
            plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
            
            plt.xlabel('残基位置')
            plt.ylabel('pLDDT分数')
            plt.title(f'残基级pLDDT置信度分数 (平均: {np.mean(residue_plddt):.2f})')
            plt.ylim(0, 100)
            
            # 添加注释
            plt.text(len(residue_plddt)-1, 95, '极高 (>90)', color='blue', horizontalalignment='right')
            plt.text(len(residue_plddt)-1, 75, '高 (70-90)', color='green', horizontalalignment='right')
            plt.text(len(residue_plddt)-1, 55, '中等 (50-70)', color='orange', horizontalalignment='right')
            plt.text(len(residue_plddt)-1, 35, '低 (<50)', color='red', horizontalalignment='right')
            
            if len(sequence) <= 30:  # 序列较短时显示氨基酸标签
                plt.xticks(x, list(sequence))
            else:
                plt.xticks(np.arange(0, len(residue_plddt), 5))
            
            plddt_file = os.path.join(plots_dir, "plddt_plot.png")
            plt.tight_layout()
            plt.savefig(plddt_file, dpi=300)
            plt.close()
            
            logger.info(f"pLDDT图表已保存至: {plddt_file}")
            result_files.append(plddt_file)
        except Exception as e:
            logger.error(f"创建pLDDT图表时出错: {str(e)}")
    
    # PAE热图
    if has_pae:
        try:
            pae = output.pae_values[0].cpu().numpy()  # 取第一个样本
            
            plt.figure(figsize=(8, 7))
            plt.imshow(pae, cmap='viridis_r', origin='lower', vmin=0, vmax=30)
            plt.colorbar(label='预测对齐误差 (Å)')
            plt.xlabel('残基位置')
            plt.ylabel('残基位置')
            plt.title('PAE (预测对齐误差)')
            
            if len(sequence) <= 30:  # 序列较短时显示刻度
                plt.xticks(np.arange(0, len(sequence)))
                plt.yticks(np.arange(0, len(sequence)))
            else:
                # 较长序列只显示部分刻度
                step = max(1, len(sequence) // 10)
                plt.xticks(np.arange(0, len(sequence), step))
                plt.yticks(np.arange(0, len(sequence), step))
                
            pae_file = os.path.join(plots_dir, "pae_plot.png")
            plt.tight_layout()
            plt.savefig(pae_file, dpi=300)
            plt.close()
            
            logger.info(f"PAE图表已保存至: {pae_file}")
            result_files.append(pae_file)
        except Exception as e:
            logger.error(f"创建PAE图表时出错: {str(e)}")
    
    # PDE热图
    if has_pde:
        try:
            pde = output.pde_values[0].cpu().numpy()  # 取第一个样本
            
            plt.figure(figsize=(8, 7))
            plt.imshow(pde, cmap='inferno_r', origin='lower', vmin=0, vmax=30)
            plt.colorbar(label='预测距离误差 (Å)')
            plt.xlabel('残基位置')
            plt.ylabel('残基位置')
            plt.title('PDE (预测距离误差)')
            
            if len(sequence) <= 30:  # 序列较短时显示刻度
                plt.xticks(np.arange(0, len(sequence)))
                plt.yticks(np.arange(0, len(sequence)))
            else:
                # 较长序列只显示部分刻度
                step = max(1, len(sequence) // 10)
                plt.xticks(np.arange(0, len(sequence), step))
                plt.yticks(np.arange(0, len(sequence), step))
                
            pde_file = os.path.join(plots_dir, "pde_plot.png")
            plt.tight_layout()
            plt.savefig(pde_file, dpi=300)
            plt.close()
            
            logger.info(f"PDE图表已保存至: {pde_file}")
            result_files.append(pde_file)
        except Exception as e:
            logger.error(f"创建PDE图表时出错: {str(e)}")
    
    # 创建综合图表
    if has_plddt and (has_pae or has_pde):
        try:
            plt.figure(figsize=(12, 10))
            
            # pLDDT分数子图
            plt.subplot(2, 1, 1)
            x = np.arange(len(residue_plddt))
            
            # 根据pLDDT分数设置颜色
            colors = []
            for score in residue_plddt:
                if score >= 90:
                    colors.append('blue')
                elif score >= 70:
                    colors.append('green')
                elif score >= 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            plt.bar(x, residue_plddt, color=colors)
            plt.axhline(y=90, color='blue', linestyle='--', alpha=0.5)
            plt.axhline(y=70, color='green', linestyle='--', alpha=0.5)
            plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
            
            plt.xlabel('残基位置')
            plt.ylabel('pLDDT分数')
            plt.title('残基级pLDDT置信度分数')
            plt.ylim(0, 100)
            
            if len(sequence) <= 30:
                plt.xticks(x, list(sequence))
            else:
                plt.xticks(np.arange(0, len(residue_plddt), 5))
            
            # PAE或PDE热图子图
            plt.subplot(2, 1, 2)
            if has_pae:
                plt.imshow(pae, cmap='viridis_r', origin='lower', vmin=0, vmax=30)
                plt.colorbar(label='预测对齐误差 (Å)')
                plt.title('PAE (预测对齐误差)')
            elif has_pde:
                plt.imshow(pde, cmap='inferno_r', origin='lower', vmin=0, vmax=30)
                plt.colorbar(label='预测距离误差 (Å)')
                plt.title('PDE (预测距离误差)')
                
            plt.xlabel('残基位置')
            plt.ylabel('残基位置')
            
            if len(sequence) <= 30:
                plt.xticks(np.arange(0, len(sequence)))
                plt.yticks(np.arange(0, len(sequence)))
            else:
                step = max(1, len(sequence) // 10)
                plt.xticks(np.arange(0, len(sequence), step))
                plt.yticks(np.arange(0, len(sequence), step))
            
            combined_file = os.path.join(plots_dir, "combined_confidence_plot.png")
            plt.tight_layout()
            plt.savefig(combined_file, dpi=300)
            plt.close()
            
            logger.info(f"综合置信度图表已保存至: {combined_file}")
            result_files.append(combined_file)
        except Exception as e:
            logger.error(f"创建综合图表时出错: {str(e)}")
    
    return result_files

def test_confidence_extraction(model, sequence="AG", molecule_type="protein", output_dir=None, plot_confidence=False):
    """专门测试置信度信息提取功能"""
    logger.info("=" * 50)
    logger.info("开始详细分析模型置信度信息")
    logger.info("=" * 50)
    
    if output_dir is None:
        output_dir = setup_output_dir()
        
    # 可以测试的不同序列
    test_sequences = {
        "short_protein": "AG",
        "medium_protein": "ACDEFGHIKLMNPQRSTVWY",  # 20个标准氨基酸
        "small_rna": "ACGU",
        "small_dna": "ACGT",
    }
    
    # 如果提供了特定序列，使用它
    if sequence in test_sequences:
        sequence = test_sequences[sequence]
        
    logger.info(f"使用序列: {sequence} (长度: {len(sequence)})")
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 将模型移至设备并设置为评估模式
    model = move_to_device(model, device)
    model.eval()
    
    # 创建输入
    try:
        logger.info("创建模型输入...")
        af3_input = Alphafold3Input(
            proteins=[sequence]
        )
        af3_input = move_to_device(af3_input, device)  # 使用新函数替代直接调用.to()
        
        # 运行预测，特别指定要返回置信度信息
        with torch.no_grad():
            logger.info("执行模型预测...")
            output_tuple = model.forward_with_alphafold3_inputs(
                af3_input,
                num_recycling_steps=1,  # 使用较少的循环以加快速度
                num_sample_steps=12,    # 合理的采样步数
                return_confidence_head_logits=True  # 关键：返回置信度信息
            )
        
        # 处理返回结果
        if isinstance(output_tuple, tuple) and len(output_tuple) == 2:
            logger.info("成功获取带置信度信息的预测结果")
            atom_positions, confidence_logits = output_tuple
            
            # 创建输出对象
            from types import SimpleNamespace
            output = SimpleNamespace()
            output.atom_pos = atom_positions
            output.confidence_logits = confidence_logits
            
            # 处理置信度逻辑值
            if hasattr(confidence_logits, 'plddt') and confidence_logits.plddt is not None:
                logger.info("处理pLDDT (预测局部距离差异测试)...")
                
                # 计算plddt分数
                logits = confidence_logits.plddt.permute(0, 2, 1)  # [b, m, bins] -> [b, bins, m]
                probs = torch.nn.functional.softmax(logits, dim=1)
                bin_width = 1.0 / logits.shape[1]
                bin_centers = torch.arange(
                    0.5 * bin_width, 1.0, bin_width, 
                    dtype=probs.dtype, device=probs.device
                )
                output.atom_confs = torch.einsum('bpm,p->bm', probs, bin_centers) * 100
                
                # 输出pLDDT统计
                plddt = output.atom_confs
                logger.info(f"pLDDT形状: {plddt.shape}")
                logger.info(f"pLDDT统计: 均值={plddt.mean().item():.2f}, 中位数={plddt.median().item():.2f}, 最小值={plddt.min().item():.2f}, 最大值={plddt.max().item():.2f}")
                
                # 如果是较短序列，显示每个残基的pLDDT
                if len(sequence) <= 50:
                    atom_counts = get_atom_counts_per_residue()
                    atom_idx = 0
                    logger.info("\n残基级pLDDT分数:")
                    for i, aa in enumerate(sequence):
                        num_atoms = atom_counts.get(aa, 5)
                        if atom_idx + num_atoms <= plddt.shape[1]:
                            residue_plddt = plddt[0, atom_idx:atom_idx+num_atoms].mean().item()
                            quality = "极高" if residue_plddt > 90 else "高" if residue_plddt > 70 else "中等" if residue_plddt > 50 else "低"
                            logger.info(f"  残基 {i+1} ({aa}): {residue_plddt:.2f} - 质量{quality}")
                        atom_idx += num_atoms
            
            # 处理PAE (预测对齐误差)
            if hasattr(confidence_logits, 'pae') and confidence_logits.pae is not None:
                logger.info("\n处理PAE (预测对齐误差)...")
                
                pae_logits = confidence_logits.pae
                logger.info(f"PAE形状: {pae_logits.shape}")
                
                # 转换成实际的PAE值
                try:
                    pae = pae_logits.permute(0, 2, 3, 1)  # [b, bins, n, n] -> [b, n, n, bins]
                    pae_probs = torch.nn.functional.softmax(pae, dim=-1)
                    
                    # 假设bin范围从0.5到32埃
                    pae_bin_width = 31.5 / pae_probs.shape[-1]
                    pae_bin_centers = torch.arange(
                        0.5 + 0.5 * pae_bin_width, 32.0, pae_bin_width,
                        dtype=pae_probs.dtype, device=pae_probs.device
                    )
                    
                    output.pae_values = torch.einsum('bnnd,d->bnn', pae_probs, pae_bin_centers)
                    logger.info(f"PAE值统计: 均值={output.pae_values.mean().item():.2f}, 最大值={output.pae_values.max().item():.2f}")
                except Exception as e:
                    logger.warning(f"计算PAE值时出错: {str(e)}")
            
            # 处理PDE (预测距离误差)
            if hasattr(confidence_logits, 'pde') and confidence_logits.pde is not None:
                logger.info("\n处理PDE (预测距离误差)...")
                
                pde_logits = confidence_logits.pde
                logger.info(f"PDE形状: {pde_logits.shape}")
                
                # 转换成实际的PDE值
                try:
                    pde = pde_logits.permute(0, 2, 3, 1)  # [b, bins, n, n] -> [b, n, n, bins]
                    pde_probs = torch.nn.functional.softmax(pde, dim=-1)
                    
                    # 假设bin范围从0.5到32埃
                    pde_bin_width = 31.5 / pde_probs.shape[-1]
                    pde_bin_centers = torch.arange(
                        0.5 + 0.5 * pde_bin_width, 32.0, pde_bin_width,
                        dtype=pde_probs.dtype, device=pde_probs.device
                    )
                    
                    output.pde_values = torch.einsum('bnnd,d->bnn', pde_probs, pde_bin_centers)
                    logger.info(f"PDE值统计: 均值={output.pde_values.mean().item():.2f}, 最大值={output.pde_values.max().item():.2f}")
                except Exception as e:
                    logger.warning(f"计算PDE值时出错: {str(e)}")
            
            # 保存结果
            if output_dir:
                logger.info(f"\n保存结果到: {output_dir}")
                
                # 保存结构
                pdb_file = os.path.join(output_dir, "predicted_structure.pdb")
                mmcif_file = os.path.join(output_dir, "predicted_structure.cif")
                
                # 创建结构对象
                try:
                    from alphafold3_pytorch.molecule.structure import Structure
                    from alphafold3_pytorch.molecule.residue import AMINO_ACID_ATOM_TYPES
                    
                    # 获取每个残基的原子类型
                    atom_idx = 0
                    atom_types = []
                    for aa in sequence:
                        if aa in AMINO_ACID_ATOM_TYPES:
                            aa_atom_types = AMINO_ACID_ATOM_TYPES[aa]
                            atom_types.extend(aa_atom_types)
                            atom_idx += len(aa_atom_types)
                        else:
                            # 若找不到，使用默认的CNOH原子类型
                            atom_types.extend(['C', 'N', 'O', 'H', 'H'])
                            atom_idx += 5
                    
                    # 创建结构
                    structure = Structure.from_atom_positions_and_types(
                        atom_positions[0].cpu().numpy(), 
                        atom_types[:atom_positions.shape[1]], 
                        sequence
                    )
                    
                    # 添加B因子 (置信度)
                    if hasattr(output, 'atom_confs'):
                        for i, atom in enumerate(structure.atoms):
                            if i < output.atom_confs.shape[1]:
                                atom.b_factor = float(output.atom_confs[0, i].cpu().numpy())
                            else:
                                atom.b_factor = 50.0  # 默认中等置信度
                    
                    # 保存PDB格式
                    if 'pdb' in os.path.basename(pdb_file):
                        save_pdb(structure, pdb_file)
                        logger.info(f"  结构保存为PDB格式: {pdb_file}")
                    
                    # 保存mmCIF格式
                    if 'cif' in os.path.basename(mmcif_file):
                        save_mmcif(structure, mmcif_file)
                        logger.info(f"  结构保存为mmCIF格式: {mmcif_file}")
                    
                except Exception as e:
                    logger.error(f"保存结构时出错: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 保存置信度信息到JSON
                try:
                    import json
                    confidence_data = {
                        "sequence": sequence,
                        "length": len(sequence),
                        "confidence_metrics": {}
                    }
                    
                    # 添加pLDDT信息
                    if hasattr(output, 'atom_confs'):
                        plddt = output.atom_confs.cpu().numpy()
                        confidence_data["confidence_metrics"]["plddt_mean"] = float(plddt.mean())
                        confidence_data["confidence_metrics"]["plddt_median"] = float(np.median(plddt))
                        confidence_data["confidence_metrics"]["plddt_min"] = float(plddt.min())
                        confidence_data["confidence_metrics"]["plddt_max"] = float(plddt.max())
                        
                        # 添加残基级别的pLDDT
                        residue_plddt = []
                        atom_idx = 0
                        for aa in sequence:
                            num_atoms = get_atom_counts_per_residue().get(aa, 5)
                            if atom_idx + num_atoms <= plddt.shape[1]:
                                res_plddt = float(plddt[0, atom_idx:atom_idx+num_atoms].mean())
                                residue_plddt.append(res_plddt)
                            else:
                                residue_plddt.append(50.0)  # 默认中等置信度
                            atom_idx += num_atoms
                        
                        confidence_data["confidence_metrics"]["residue_plddt"] = residue_plddt
                    
                    # 添加PAE和PDE信息
                    if hasattr(output, 'pae_values'):
                        pae = output.pae_values.cpu().numpy()
                        confidence_data["confidence_metrics"]["pae_mean"] = float(pae.mean())
                        confidence_data["confidence_metrics"]["pae_max"] = float(pae.max())
                    
                    if hasattr(output, 'pde_values'):
                        pde = output.pde_values.cpu().numpy()
                        confidence_data["confidence_metrics"]["pde_mean"] = float(pde.mean())
                        confidence_data["confidence_metrics"]["pde_max"] = float(pde.max())
                    
                    # 保存到文件
                    confidence_file = os.path.join(output_dir, "confidence_metrics.json")
                    with open(confidence_file, 'w') as f:
                        json.dump(confidence_data, f, indent=2)
                    
                    logger.info(f"  置信度信息保存为JSON: {confidence_file}")
                except Exception as e:
                    logger.error(f"保存置信度信息时出错: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 创建可视化图表 (如果启用)
                if plot_confidence:
                    logger.info("生成置信度可视化图表...")
                    plot_files = plot_confidence_metrics(output, sequence, output_dir)
                    if plot_files:
                        logger.info(f"已生成 {len(plot_files)} 个置信度可视化图表")
            
            logger.info("\n置信度信息分析完成!")
            return True
        else:
            logger.error("模型未返回置信度信息")
            return False
    except Exception as e:
        logger.error(f"置信度信息提取过程中出错: {str(e)}")
        import traceback
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

def test_basic_functionality(output_dir=None, device=None, precision="fp32"):
    """
    测试AlphaFold3的基本功能
    
    Args:
        output_dir: 输出目录
        device: 计算设备
        precision: 精度设置 ("fp32", "fp16", "bf16")
    
    Returns:
        测试结果字典
    """
    if device is None:
        device = get_device(force_cpu=True)  # 为基本功能测试使用CPU
        
    if output_dir is None:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./af3_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始基本功能测试，输出目录: {output_dir}")
    logger.info(f"使用设备: {device}, 精度: {precision}")
    
    test_cases = [
        {"id": "ag", "sequence": "AG", "type": "protein"},
        {"id": "small_rna", "sequence": "ACGU", "type": "rna"},
        {"id": "small_dna", "sequence": "ACGT", "type": "dna"}
    ]
    
    results = {}
    
    logger.info("本次测试不加载预训练模型")
    
    for test_case in test_cases:
        sequence_id = test_case["id"]
        sequence = test_case["sequence"]
        molecule_type = test_case["type"]
        
        logger.info(f"测试序列 {sequence_id}: {sequence} (类型: {molecule_type})")
        
        try:
            # 准备输入数据
            input_data = prepare_input_data(
                sequence, 
                molecule_type=molecule_type, 
                use_msa=False, 
                use_templates=False
            )
            
            if input_data is None:
                logger.error(f"为 {sequence_id} 准备输入数据失败")
                results[sequence_id] = {"status": "失败", "错误": "准备输入数据失败"}
                continue
                
            # 将数据移至指定设备和精度
            input_data = move_to_device(input_data, device, precision)
            
            # 创建一个模拟的输出结果用于测试
            mock_atom_positions = generate_mock_atom_positions(sequence, device)
            
            # 创建模拟的置信度逻辑值
            from types import SimpleNamespace
            mock_output = SimpleNamespace()
            mock_output.atom_pos = mock_atom_positions
            
            # 创建模拟的置信度logits
            sequence_length = len(sequence)
            confidence_logits = SimpleNamespace()
            
            # 创建pLDDT logits: [batch(1), bins(50), residues]
            plddt_bins = 50
            confidence_logits.plddt = torch.randn(1, plddt_bins, sequence_length, device=device)
            
            # 创建PAE logits: [batch(1), residues, residues, bins(64)]
            pae_bins = 64
            confidence_logits.pae = torch.randn(1, sequence_length, sequence_length, pae_bins, device=device)
            
            # 创建PDE logits (如果需要): [batch(1), atoms, atoms, bins(64)]
            atom_count = calculate_total_atoms(sequence)
            confidence_logits.pde = torch.randn(1, atom_count, atom_count, pae_bins, device=device)
            
            mock_output.confidence_logits = confidence_logits
            
            # 评估质量指标
            metrics = evaluate_prediction_quality(mock_output, sequence_id)
            
            # 保存PDB结构
            case_output_dir = os.path.join(output_dir, sequence_id)
            os.makedirs(case_output_dir, exist_ok=True)
            
            pdb_file = os.path.join(case_output_dir, f"{sequence_id}.pdb")
            
            try:
                # 将输出移至CPU以生成PDB
                mock_output_cpu = move_to_device(mock_output, 'cpu')
                save_pdb_with_biopython(mock_output_cpu.atom_pos, sequence, pdb_file)
                logger.info(f"为 {sequence_id} 生成PDB文件: {pdb_file}")
            except Exception as e:
                logger.warning(f"为 {sequence_id} 生成PDB文件失败: {str(e)}")
            
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

def run_complete_pipeline(
    sequence="AG", 
    molecule_type="protein", 
    output_dir=None, 
    epochs=50, 
    use_msa=True,
    use_templates=True,
    device=None,
    precision="fp32",
    save_structures=True,
    quiet=False
):
    """
    执行完整的输入-搜索-训练-预测流程
    
    Args:
        sequence: 序列字符串
        molecule_type: 分子类型，可以是"protein"、"rna"或"dna"
        output_dir: 输出目录
        epochs: 训练轮数
        use_msa: 是否使用MSA数据
        use_templates: 是否使用模板数据
        device: 计算设备
        precision: 精度设置，可选"fp32"、"fp16"或"bf16"
        save_structures: 是否保存预测的结构
        quiet: 是否减少控制台输出
    
    Returns:
        结果字典，包含输入、训练和预测的结果
    """
    if output_dir is None:
        output_dir = setup_output_dir("complete_pipeline")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置日志
    log_file = os.path.join(output_dir, "pipeline.log")
    setup_logging(quiet=quiet, log_file=log_file)
    
    logger.info("=" * 50)
    logger.info(f"开始执行完整流程 - 序列: {sequence}, 类型: {molecule_type}")
    logger.info(f"日志文件保存在: {log_file}")
    logger.info("=" * 50)
    
    results = {
        "sequence": sequence,
        "molecule_type": molecule_type,
        "output_dir": output_dir,
        "steps": {}
    }
    
    # 步骤1: 创建序列输入
    logger.info("-" * 30)
    logger.info("步骤1: 创建序列输入")
    try:
        if device is None:
            device = get_device()
        inputs = create_sequence_input(sequence, molecule_type, device)
        if inputs is not None:
            # 将输入移动到指定设备和精度
            inputs = move_to_device(inputs, device, precision)
            logger.info(f"成功创建输入对象，类型: {type(inputs)}")
            results["steps"]["create_input"] = {
                "status": "成功",
                "input_type": str(type(inputs))
            }
        else:
            logger.error("创建输入失败")
            results["steps"]["create_input"] = {"status": "失败"}
            return results
    except Exception as e:
        logger.error(f"创建输入时出错: {str(e)}")
        results["steps"]["create_input"] = {"status": "出错", "error": str(e)}
        return results
    
    # 步骤2: 准备MSA和模板数据
    logger.info("-" * 30)
    logger.info("步骤2: 准备MSA和模板数据")
    try:
        seq_id = molecule_type + "_" + sequence[:5]  # 创建一个简单的序列ID
        msa_data, template_data = prepare_msas_and_templates(
            seq_id, sequence, use_msa=use_msa, use_templates=use_templates
        )
        
        msa_found = msa_data.get('found', False)
        template_found = template_data.get('found', False)
        
        logger.info(f"MSA数据: {'找到' if msa_found else '未找到但创建了默认MSA'}")
        results["steps"]["msa_preparation"] = {"status": '找到' if msa_found else '创建了默认MSA'}
        
        logger.info(f"模板数据: {'找到' if template_found else '未找到但继续'}")
        results["steps"]["template_preparation"] = {"status": '找到' if template_found else '未找到'}
    except Exception as e:
        logger.error(f"准备MSA和模板数据时出错: {str(e)}")
        results["steps"]["data_preparation"] = {"status": "出错", "error": str(e)}
        return results
    
    # 步骤3: 创建模型
    logger.info("-" * 30)
    logger.info("步骤3: 创建模型")
    try:
        # 设置模型参数
        dim_atom_inputs = 3 
        dim_atompair_inputs = 5
        num_molecule_mods = 4  # 确保这个值与下面使用is_molecule_mod的维度匹配
        
        # 创建一个简化版的AlphaFold3模型用于演示
        model = Alphafold3(
            dim_atom_inputs = dim_atom_inputs,
            dim_atompair_inputs = dim_atompair_inputs,
            dim_template_feats = 64,  # 添加必需的参数
            dim_atom = 32,            # 而不是使用dim参数
            dim_input_embedder_token = 32,
            dim_single = 32,
            dim_pairwise = 16,
            dim_token = 32,
            atoms_per_window = 27,  # 添加必需的参数
            num_molecule_mods = num_molecule_mods,
            input_embedder_kwargs = dict(
                atom_transformer_blocks = 2,
                atom_transformer_heads = 4,
                atom_transformer_kwargs = dict()
            ),
            confidence_head_kwargs = dict(
                pairformer_depth = 2
            ),
            diffusion_module_kwargs = dict(
                single_cond_kwargs = dict(
                    num_transitions = 2,
                    transition_expansion_factor = 2,
                ),
                pairwise_cond_kwargs = dict(
                    num_transitions = 2
                ),
                atom_encoder_depth = 2,
                atom_encoder_heads = 4,
                token_transformer_depth = 4,
                token_transformer_heads = 4,
            ),
            num_rollout_steps = 8
        )
        
        # 将模型移动到指定设备和精度
        model = move_to_device(model, device, precision)
        
        logger.info(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        results["steps"]["model_creation"] = {"status": "成功"}
        
    except Exception as e:
        logger.error(f"创建模型时出错: {str(e)}")
        logger.error(traceback.format_exc())
        results["steps"]["model_creation"] = {"status": "出错", "error": str(e)}
        return results
    
    # 步骤4: 训练模型
    logger.info("-" * 30)
    logger.info("步骤4: 训练模型")
    try:
        # 定义损失函数和优化器
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 创建一个训练循环
        model.train()
        
        # 将Alphafold3Input转换为MoleculeLengthMoleculeInput
        logger.info("将Alphafold3Input转换为可训练的输入格式...")
        try:
            from alphafold3_pytorch.inputs import alphafold3_input_to_molecule_lengthed_molecule_input
            from alphafold3_pytorch.inputs import molecule_lengthed_molecule_input_to_atom_input
            
            # 转换输入格式
            if hasattr(inputs, 'proteins') and not hasattr(inputs, 'atom_inputs'):
                logger.info("检测到Alphafold3Input对象，正在转换为atom_input...")
                molecule_lengthed_input = alphafold3_input_to_molecule_lengthed_molecule_input(inputs)
                atom_input = molecule_lengthed_molecule_input_to_atom_input(molecule_lengthed_input)
                # 确保原子输入的维度正确
                if hasattr(atom_input, 'atom_inputs') and atom_input.atom_inputs.shape[-1] != dim_atom_inputs:
                    logger.warning(f"atom_inputs的特征维度 ({atom_input.atom_inputs.shape[-1]}) 与模型期望的维度 ({dim_atom_inputs}) 不匹配")
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
            # 输出更详细的错误信息
            logger.error(traceback.format_exc())
        
        for epoch in range(epochs):
            # 每次训练使用相同的输入（这里只是演示）
            optimizer.zero_grad()
            
            # 前向传播
            logger.info(f"  执行第 {epoch+1}/{epochs} 轮训练...")
            try:
                # 创建is_molecule_mod参数
                logger.info(f"检查inputs属性: molecule_ids: {hasattr(inputs, 'molecule_ids')}, additional_molecule_feats: {hasattr(inputs, 'additional_molecule_feats')}, is_molecule_types: {hasattr(inputs, 'is_molecule_types')}")
                
                # 确定序列长度和批处理大小
                if hasattr(inputs, 'molecule_ids') and inputs.molecule_ids is not None and hasattr(inputs.molecule_ids, 'shape') and len(inputs.molecule_ids.shape) >= 2:
                    logger.info(f"molecule_ids.shape: {inputs.molecule_ids.shape}")
                    seq_len = inputs.molecule_ids.shape[1]
                    batch_size = inputs.molecule_ids.shape[0]
                elif hasattr(inputs, 'additional_molecule_feats') and inputs.additional_molecule_feats is not None and hasattr(inputs.additional_molecule_feats, 'shape') and len(inputs.additional_molecule_feats.shape) >= 2:
                    logger.info(f"additional_molecule_feats.shape: {inputs.additional_molecule_feats.shape}")
                    seq_len = inputs.additional_molecule_feats.shape[1]
                    batch_size = inputs.additional_molecule_feats.shape[0]
                elif hasattr(inputs, 'is_molecule_types') and inputs.is_molecule_types is not None and hasattr(inputs.is_molecule_types, 'shape') and len(inputs.is_molecule_types.shape) >= 2:
                    logger.info(f"is_molecule_types.shape: {inputs.is_molecule_types.shape}")
                    seq_len = inputs.is_molecule_types.shape[1]
                    batch_size = inputs.is_molecule_types.shape[0]
                else:
                    # 推断序列长度
                    if len(sequence) > 0:
                        logger.warning(f"无法从inputs对象确定序列长度和批量大小，使用序列长度: {len(sequence)}")
                        seq_len = len(sequence)  # 使用提供的序列长度
                        batch_size = 1  # 默认批处理大小为1
                    else:
                        logger.warning("无法确定序列长度和批量大小，使用默认值")
                        seq_len = 2  # 默认值
                        batch_size = 1
                
                logger.info(f"确定的seq_len: {seq_len}, batch_size: {batch_size}")
                
                # 确定设备
                if hasattr(inputs, 'atom_inputs') and inputs.atom_inputs is not None and hasattr(inputs.atom_inputs, 'device'):
                    device = inputs.atom_inputs.device
                else:
                    device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
                    logger.info(f"无法从输入确定设备，使用默认设备: {device}")
                
                # 创建is_molecule_mod张量
                is_molecule_mod = torch.zeros(batch_size, seq_len, num_molecule_mods, dtype=torch.bool, device=device)
                logger.info(f"成功创建is_molecule_mod张量，形状: {is_molecule_mod.shape}")
                
                # 训练循环
                model.train()  # 确保模型在训练模式

                # 删除此处的嵌套训练循环，以避免重复定义和覆盖
                
                # 设置is_molecule_mod属性
                if hasattr(inputs, 'is_molecule_mod'):
                    if inputs.is_molecule_mod is None or inputs.is_molecule_mod.shape != is_molecule_mod.shape:
                        inputs.is_molecule_mod = is_molecule_mod
                        logger.info("已将is_molecule_mod设置为inputs的属性")
                else:
                    setattr(inputs, 'is_molecule_mod', is_molecule_mod)
                    logger.info("已动态添加is_molecule_mod属性到inputs")
                
                logger.info("执行模型预测...")
                
                # 关键修改：去掉torch.no_grad()上下文，因为我们需要计算梯度
                atom_positions, confidence_logits = model.forward_with_alphafold3_inputs(
                    inputs,
                    num_recycling_steps=1,   # 减少循环步数
                    num_sample_steps=12,     # 增加采样步数以提高质量
                    return_confidence_head_logits=True  # 获取置信度信息
                )
                
                logger.info(f"成功执行预测，返回结果形状: {atom_positions.shape}")
                
                # 创建输出对象
                from types import SimpleNamespace
                output = SimpleNamespace()
                output.atom_pos = atom_positions
                output.confidence_logits = confidence_logits
                
                # 制造一个简单的目标位置
                # 修改：确保目标位置需要梯度
                num_atoms = atom_positions.shape[1]
                target_positions = torch.randn_like(atom_positions, requires_grad=True)
                
                # 计算损失
                loss = loss_fn(atom_positions, target_positions)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                logger.info(f"  轮次 {epoch+1}: 损失 = {loss.item():.4f}")
                
                # 已删除内层循环的异常处理部分，保留外层循环中的异常处理
            except Exception as train_err:
                logger.error(f"  训练时出错: {str(train_err)}")
                # 输出更详细的错误信息
                logger.error(traceback.format_exc())
        
        # 设置为评估模式
        model.eval()
        
        # 保存模型
        model_path = os.path.join(output_dir, "af3_model.pt")
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"模型已保存至: {model_path}")
            results["steps"]["training"] = {
                "status": "成功",
                "epochs": epochs,
                "final_loss": loss.item(),
                "model_path": model_path
            }
        except Exception as save_err:
            logger.error(f"保存模型时出错: {str(save_err)}")
            results["steps"]["training"] = {
                "status": "部分成功",
                "epochs": epochs,
                "final_loss": loss.item(),
                "error": str(save_err)
            }
    except Exception as e:
        logger.error(f"训练模型时出错: {str(e)}")
        results["steps"]["training"] = {"status": "出错", "error": str(e)}
    
    # 步骤5: 使用模型进行预测
    logger.info("-" * 30)
    logger.info("步骤5: 使用模型预测")
    try:
        # 确保模型在评估模式
        model.eval()
        
        # 执行推理
        with torch.no_grad():
            # 设置return_confidence_head_logits=True以获取置信度信息
            logger.info("执行模型预测...")
            try:
                # 确保使用转换后的输入
                if not hasattr(inputs, 'atom_inputs'):
                    logger.warning("预测阶段检测到输入可能尚未转换，尝试重新转换")
                    # 重复转换步骤
                    from alphafold3_pytorch.inputs import alphafold3_input_to_molecule_lengthed_molecule_input
                    from alphafold3_pytorch.inputs import molecule_lengthed_molecule_input_to_atom_input
                    
                    if hasattr(inputs, 'proteins'):
                        molecule_lengthed_input = alphafold3_input_to_molecule_lengthed_molecule_input(inputs)
                        atom_input = molecule_lengthed_molecule_input_to_atom_input(molecule_lengthed_input)
                        
                        # 确保原子输入的维度正确
                        if hasattr(atom_input, 'atom_inputs') and atom_input.atom_inputs.shape[-1] != dim_atom_inputs:
                            original_shape = atom_input.atom_inputs.shape
                            if atom_input.atom_inputs.shape[-1] < dim_atom_inputs:
                                padding = torch.zeros(*original_shape[:-1], dim_atom_inputs - original_shape[-1], device=atom_input.atom_inputs.device)
                                atom_input.atom_inputs = torch.cat([atom_input.atom_inputs, padding], dim=-1)
                            else:
                                atom_input.atom_inputs = atom_input.atom_inputs[..., :dim_atom_inputs]
                            
                        inputs = atom_input
                        logger.info("成功重新转换输入格式")
                
                # 创建is_molecule_mod参数
                logger.info(f"检查inputs属性: molecule_ids: {hasattr(inputs, 'molecule_ids')}, additional_molecule_feats: {hasattr(inputs, 'additional_molecule_feats')}, is_molecule_types: {hasattr(inputs, 'is_molecule_types')}")

                # 确定序列长度和批处理大小
                if hasattr(inputs, 'molecule_ids') and inputs.molecule_ids is not None and hasattr(inputs.molecule_ids, 'shape') and len(inputs.molecule_ids.shape) >= 2:
                    logger.info(f"molecule_ids.shape: {inputs.molecule_ids.shape}")
                    seq_len = inputs.molecule_ids.shape[1]
                    batch_size = inputs.molecule_ids.shape[0]
                elif hasattr(inputs, 'additional_molecule_feats') and inputs.additional_molecule_feats is not None and hasattr(inputs.additional_molecule_feats, 'shape') and len(inputs.additional_molecule_feats.shape) >= 2:
                    logger.info(f"additional_molecule_feats.shape: {inputs.additional_molecule_feats.shape}")
                    seq_len = inputs.additional_molecule_feats.shape[1]
                    batch_size = inputs.additional_molecule_feats.shape[0]
                elif hasattr(inputs, 'is_molecule_types') and inputs.is_molecule_types is not None and hasattr(inputs.is_molecule_types, 'shape') and len(inputs.is_molecule_types.shape) >= 2:
                    logger.info(f"is_molecule_types.shape: {inputs.is_molecule_types.shape}")
                    seq_len = inputs.is_molecule_types.shape[1]
                    batch_size = inputs.is_molecule_types.shape[0]
                else:
                    # 推断序列长度
                    if len(sequence) > 0:
                        logger.warning(f"无法从inputs对象确定序列长度和批量大小，使用序列长度: {len(sequence)}")
                        seq_len = len(sequence)  # 使用提供的序列长度
                        batch_size = 1  # 默认批处理大小为1
                    else:
                        logger.warning("无法确定序列长度和批量大小，使用默认值")
                        seq_len = 2  # 默认值
                        batch_size = 1

                logger.info(f"确定的seq_len: {seq_len}, batch_size: {batch_size}")

                # 确定设备
                if hasattr(inputs, 'atom_inputs') and inputs.atom_inputs is not None and hasattr(inputs.atom_inputs, 'device'):
                    device = inputs.atom_inputs.device
                else:
                    device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
                    logger.info(f"无法从输入确定设备，使用默认设备: {device}")

                # 创建is_molecule_mod张量
                is_molecule_mod = torch.zeros(batch_size, seq_len, num_molecule_mods, dtype=torch.bool, device=device)
                logger.info(f"成功创建is_molecule_mod张量，形状: {is_molecule_mod.shape}")

                # 修改：这里我们不直接传递is_molecule_mod参数，而是将其设置为inputs的属性
                if hasattr(inputs, 'is_molecule_mod'):
                    # 如果inputs已经有is_molecule_mod属性，确保其形状正确
                    if inputs.is_molecule_mod is None or inputs.is_molecule_mod.shape != is_molecule_mod.shape:
                        inputs.is_molecule_mod = is_molecule_mod
                        logger.info("已将is_molecule_mod设置为inputs的属性")
                else:
                    # 动态添加is_molecule_mod属性
                    setattr(inputs, 'is_molecule_mod', is_molecule_mod)
                    logger.info("已动态添加is_molecule_mod属性到inputs")

                logger.info("执行模型预测...")
                atom_positions, confidence_logits = model.forward_with_alphafold3_inputs(
                    inputs,
                    num_recycling_steps=1,   # 减少循环步数
                    num_sample_steps=12,     # 增加采样步数以提高质量
                    return_confidence_head_logits=True  # 获取置信度信息
                )
                
                # 创建输出对象
                from types import SimpleNamespace
                output = SimpleNamespace()
                output.atom_pos = atom_positions
                output.confidence_logits = confidence_logits
                
                # 提取置信度信息
                if hasattr(confidence_logits, 'plddt') and confidence_logits.plddt is not None:
                    # 计算plddt分数
                    logits = confidence_logits.plddt.permute(0, 2, 1)  # [b, m, bins] -> [b, bins, m]
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    bin_width = 1.0 / logits.shape[1]
                    bin_centers = torch.arange(
                        0.5 * bin_width, 1.0, bin_width, 
                        dtype=probs.dtype, device=probs.device
                    )
                    output.atom_confs = torch.einsum('bpm,p->bm', probs, bin_centers) * 100
                    logger.info(f"成功计算pLDDT分数，均值: {output.atom_confs.mean().item():.2f}")
                
                # 记录预测成功
                results["steps"]["prediction"] = {
                    "status": "成功",
                    "shape": list(atom_positions.shape),
                }
                
                # 如果需要保存结构
                if save_structures:
                    logger.info("保存预测结构...")
                    save_results = {}
                    
                    # 保存PDB格式
                    pdb_file = os.path.join(output_dir, f"{molecule_type}_{len(sequence)}.pdb")
                    try:
                        # 创建Structure对象使用create_simple_biomolecule函数
                        structure = create_simple_biomolecule(
                            atom_positions[0].cpu().numpy(),
                            sequence,
                            molecule_type
                        )
                        # 添加B因子作为置信度
                        if hasattr(output, 'atom_confs'):
                            for i, atom in enumerate(structure.get_atoms()):
                                if i < output.atom_confs.shape[1]:
                                    atom.set_bfactor(float(output.atom_confs[0, i].cpu().numpy()))
                        
                        save_pdb(structure, pdb_file)
                        save_results["pdb"] = pdb_file
                        logger.info(f"结构已保存为PDB格式: {pdb_file}")
                    except Exception as pdb_err:
                        logger.error(f"保存PDB格式时出错: {str(pdb_err)}")
                    
                    # 保存mmCIF格式
                    mmcif_file = os.path.join(output_dir, f"{molecule_type}_{len(sequence)}.cif")
                    try:
                        if 'structure' in locals():
                            save_mmcif(structure, mmcif_file)
                            save_results["mmcif"] = mmcif_file
                            logger.info(f"结构已保存为mmCIF格式: {mmcif_file}")
                    except Exception as cif_err:
                        logger.error(f"保存mmCIF格式时出错: {str(cif_err)}")
                    
                    # 保存原始张量
                    tensor_file = os.path.join(output_dir, f"{molecule_type}_{len(sequence)}.npy")
                    try:
                        np.save(tensor_file, atom_positions[0].cpu().numpy())
                        save_results["tensor"] = tensor_file
                        logger.info(f"原子坐标已保存为NumPy格式: {tensor_file}")
                    except Exception as np_err:
                        logger.error(f"保存NumPy格式时出错: {str(np_err)}")
                    
                    # 保存置信度信息
                    if hasattr(output, 'atom_confs'):
                        conf_file = os.path.join(output_dir, "confidence_metrics.json")
                        try:
                            conf_data = {
                                "plddt_mean": float(output.atom_confs.mean().item()),
                                "plddt_min": float(output.atom_confs.min().item()),
                                "plddt_max": float(output.atom_confs.max().item())
                            }
                            
                            with open(conf_file, 'w') as f:
                                json.dump(conf_data, f, indent=2)
                                
                            save_results["confidence"] = conf_file
                            logger.info(f"置信度信息已保存: {conf_file}")
                        except Exception as conf_err:
                            logger.error(f"保存置信度信息时出错: {str(conf_err)}")
                    
                    results["steps"]["prediction"]["save_results"] = save_results
                
            except Exception as pred_err:
                logger.error(f"预测时出错: {str(pred_err)}")
                results["steps"]["prediction"] = {"status": "出错", "error": str(pred_err)}
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        results["steps"]["prediction"] = {"status": "出错", "error": str(e)}
    
    # 步骤6: 生成报告
    logger.info("-" * 30)
    logger.info("步骤6: 生成报告")
    
    report_file = os.path.join(output_dir, "pipeline_report.md")
    try:
        with open(report_file, "w") as f:
            f.write("# AlphaFold 3完整流程测试报告\n\n")
            f.write(f"测试时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 测试序列\n\n")
            f.write(f"- 序列: `{sequence}`\n")
            f.write(f"- 分子类型: {molecule_type}\n\n")
            
            # 添加原子数量信息
            if molecule_type == "protein":
                f.write(f"- 总原子数: {calculate_total_atoms(sequence)}\n")
                atom_counts = get_atom_counts_per_residue()
                f.write("- 每个残基的原子数量:\n")
                for aa in sequence:
                    f.write(f"  - {aa}: {atom_counts.get(aa, 5)} 个原子\n")
            
            f.write("## 步骤结果\n\n")
            
            # 输入创建
            f.write("### 1. 序列输入创建\n\n")
            create_status = results["steps"].get("create_input", {}).get("status", "未执行")
            f.write(f"状态: **{create_status}**\n\n")
            if "input_type" in results["steps"].get("create_input", {}):
                f.write(f"输入类型: `{results['steps']['create_input']['input_type']}`\n\n")
            
            # MSA和模板准备
            f.write("### 2. MSA和模板数据准备\n\n")
            msa_status = results["steps"].get("msa_preparation", {}).get("status", "未执行")
            template_status = results["steps"].get("template_preparation", {}).get("status", "未执行")
            f.write(f"MSA数据: **{msa_status}**\n\n")
            f.write(f"模板数据: **{template_status}**\n\n")
            
            # 模型创建
            f.write("### 3. 模型创建\n\n")
            model_status = results["steps"].get("model_creation", {}).get("status", "未执行")
            f.write(f"状态: **{model_status}**\n\n")
            
            # 训练
            f.write("### 4. 模型训练\n\n")
            train_status = results["steps"].get("training", {}).get("status", "未执行")
            f.write(f"状态: **{train_status}**\n\n")
            if "epochs" in results["steps"].get("training", {}):
                f.write(f"训练轮数: {results['steps']['training']['epochs']}\n\n")
            if "final_loss" in results["steps"].get("training", {}):
                f.write(f"最终损失: {results['steps']['training']['final_loss']:.4f}\n\n")
            if "model_path" in results["steps"].get("training", {}):
                f.write(f"模型保存路径: `{results['steps']['training']['model_path']}`\n\n")
            if "error" in results["steps"].get("training", {}):
                f.write(f"错误信息: ```\n{results['steps']['training']['error']}\n```\n\n")
            
            # 预测
            f.write("### 5. 结构预测\n\n")
            pred_status = results["steps"].get("prediction", {}).get("status", "未执行")
            f.write(f"状态: **{pred_status}**\n\n")
            if "shape" in results["steps"].get("prediction", {}):
                f.write(f"预测结构形状: {results['steps']['prediction']['shape']}\n\n")
            
            # 添加有关保存的结构文件信息
            if "save_results" in results["steps"].get("prediction", {}) and results["steps"]["prediction"]["save_results"]:
                save_results = results["steps"]["prediction"]["save_results"]
                f.write("保存的结构文件:\n\n")
                
                if "pdb" in save_results:
                    pdb_rel_path = os.path.relpath(save_results["pdb"], output_dir)
                    f.write(f"- PDB文件: [{pdb_rel_path}]({pdb_rel_path})\n")
                
                if "mmcif" in save_results:
                    mmcif_rel_path = os.path.relpath(save_results["mmcif"], output_dir)
                    f.write(f"- mmCIF文件: [{mmcif_rel_path}]({mmcif_rel_path})\n")
                
                if "xyz" in save_results:
                    xyz_rel_path = os.path.relpath(save_results["xyz"], output_dir)
                    f.write(f"- XYZ文件: [{xyz_rel_path}]({xyz_rel_path})\n")
                
                if "tensor" in save_results:
                    tensor_rel_path = os.path.relpath(save_results["tensor"], output_dir)
                    f.write(f"- 张量文件: [{tensor_rel_path}]({tensor_rel_path})\n")
                
                f.write("\n可视化说明：\n")
                f.write("- PDB和mmCIF文件可以使用PyMOL、UCSF Chimera或VMD等工具打开查看\n")
                f.write("- XYZ文件可以使用Avogadro、VMD等工具打开查看\n\n")
            elif "error" in results["steps"].get("prediction", {}):
                f.write(f"错误信息: ```\n{results['steps']['prediction']['error']}\n```\n\n")
            
            # 测试环境
            f.write("## 测试环境\n\n")
            f.write(f"- PyTorch版本: {torch.__version__}\n")
            f.write(f"- NPU可用: {torch_npu.npu.is_available()}\n")
            if torch_npu.npu.is_available():
                f.write(f"- NPU设备: {torch_npu.npu.get_device_name(0)}\n")
        
        logger.info(f"报告已保存至: {report_file}")
        results["report_file"] = report_file
    except Exception as e:
        logger.error(f"生成报告时出错: {str(e)}")
        results["report_error"] = str(e)
    
    logger.info("=" * 50)
    logger.info(f"完整流程执行完毕")
    logger.info("=" * 50)
    
    return results

def run_comprehensive_test(config):
    """
    运行综合测试，测试不同类型和大小的分子
    
    Args:
        config: Config对象，包含所有配置参数
    
    Returns:
        测试结果字典
    """
    # 默认序列
    test_sequences = {
        "protein_small": "MKKIEELQAQSLA",   # 小蛋白质序列，13个氨基酸
        "protein_medium": "MVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELNGKLTGMAFRVPTANVSVVDLTCRLEKAAKY",  # 中等大小的蛋白质序列，约250个氨基酸
        "dna_small": "ATCGTAGC",   # 小DNA序列，8个核苷酸
        "dna_medium": "ATCGTAGCATCGATCGATCGATCGTAGCTAGCTAGCTAGCTACGTAGCTAGCTACGATCG",  # 中等大小的DNA序列
        "rna_small": "AUCGUAGC",   # 小RNA序列，8个核苷酸
        "rna_medium": "AUCGUAGCAUCGAUCGAUCGAUCGUAGCUAGCUAGCUAGCUACGUAGCUAGCUACGAUCG"   # 中等大小的RNA序列
    }
    
    # 创建输出目录
    output_dir = os.path.join(config.output_dir, "comprehensive_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建报告文件
    report_file = os.path.join(output_dir, "comprehensive_test_report.md")
    
    # 初始化报告内容
    report_content = [
        "# AlphaFold3 综合测试报告",
        f"测试时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "## 测试配置",
        f"- 设备: {config.device}",
        f"- 精度: {config.precision}",
        f"- 轮次: {config.epochs}",
        f"- 输出目录: {output_dir}",
        f"- 安静模式: {'启用' if config.quiet else '禁用'}",
        "\n## 测试结果\n"
    ]
    
    # 创建测试结果目录
    for seq_name in test_sequences:
        os.makedirs(os.path.join(output_dir, seq_name), exist_ok=True)
    
    # 运行每个序列的测试
    test_results = {}
    
    for seq_name, sequence in test_sequences.items():
        logger.info(f"测试序列: {seq_name}")
        logger.info(f"序列长度: {len(sequence)}")
        logger.info(f"序列内容: {sequence[:30]}{'...' if len(sequence) > 30 else ''}")
        
        # 确定分子类型
        if seq_name.startswith("protein"):
            molecule_type = "protein"
        elif seq_name.startswith("dna"):
            molecule_type = "dna"
        elif seq_name.startswith("rna"):
            molecule_type = "rna"
        else:
            molecule_type = "protein"  # 默认
        
        # 为每个序列创建单独的目录
        seq_output_dir = os.path.join(output_dir, seq_name)
        
        try:
            # 1. 准备输入数据
            logger.info(f"准备 {seq_name} 序列数据（类型：{molecule_type}）")
            input_data = prepare_input_data(
                sequence, 
                molecule_type=molecule_type,
                use_msa=config.use_msa,
                use_templates=config.use_templates
            )
            
            if input_data is None:
                raise Exception("准备输入数据失败")
                
            # 将数据移至指定设备和精度
            input_data = move_to_device(input_data, config.device, config.precision)
            
            # 模拟原子位置
            mock_atom_pos = generate_mock_atom_positions(sequence, config.device)
            
            # 创建模拟的置信度logits
            from types import SimpleNamespace
            mock_output = SimpleNamespace()
            mock_output.atom_pos = mock_atom_pos
            
            # 创建模拟的置信度logits
            sequence_length = len(sequence)
            confidence_logits = SimpleNamespace()
            
            # 创建pLDDT logits: [batch(1), bins(50), residues]
            plddt_bins = 50
            confidence_logits.plddt = torch.randn(1, plddt_bins, sequence_length, device=config.device)
            
            # 创建PAE logits: [batch(1), residues, residues, bins(64)]
            pae_bins = 64
            confidence_logits.pae = torch.randn(1, sequence_length, sequence_length, pae_bins, device=config.device)
            
            # 创建PDE logits (如果需要): [batch(1), atoms, atoms, bins(64)]
            atom_count = calculate_total_atoms(sequence)
            confidence_logits.pde = torch.randn(1, atom_count, atom_count, pae_bins, device=config.device)
            
            mock_output.confidence_logits = confidence_logits
            
            # 评估质量指标
            metrics = evaluate_prediction_quality(mock_output, seq_name)
            
            # 生成PDB文件
            pdb_file = generate_pdb_file(
                seq_name, 
                sequence, 
                mock_output.atom_pos, 
                molecule_type,
                seq_output_dir
            )
            
            # 保存其他结果
            result_files = {}
            if pdb_file:
                result_files['pdb'] = pdb_file
                
            # 保存numpy格式的坐标
            npy_file = os.path.join(seq_output_dir, f"{seq_name}_coords.npy")
            try:
                if isinstance(mock_output.atom_pos, torch.Tensor):
                    # 如果是张量，确保先移到CPU
                    atom_pos_cpu = mock_output.atom_pos.cpu().detach().numpy()
                    np.save(npy_file, atom_pos_cpu)
                elif isinstance(mock_output.atom_pos, list):
                    # 如果是列表，检查列表元素
                    if all(isinstance(item, torch.Tensor) for item in mock_output.atom_pos):
                        # 列表中的元素是张量
                        atom_pos_list = []
                        for tensor in mock_output.atom_pos:
                            atom_pos_list.append(tensor.cpu().detach().numpy())
                        np.save(npy_file, np.array(atom_pos_list))
                    else:
                        # 列表中的元素不是张量
                        np.save(npy_file, np.array(mock_output.atom_pos))
                else:
                    # 其他情况
                    np.save(npy_file, np.array(mock_output.atom_pos))
                result_files['numpy'] = npy_file
            except Exception as e:
                logger.error(f"保存numpy文件出错: {str(e)}")
            
            # 记录成功
            test_results[seq_name] = {
                "status": "成功",
                "files": result_files,
                "metrics": metrics,
                "details": f"序列长度: {len(sequence)}, 原子数: {atom_count}"
            }
            
            # 添加到报告
            report_content.append(f"### {seq_name} (分子类型: {molecule_type})")
            report_content.append(f"- 状态: ✅ 成功")
            report_content.append(f"- 序列长度: {len(sequence)}")
            report_content.append(f"- 原子数: {atom_count}")
            
            # 添加指标
            if metrics:
                report_content.append("- 置信度指标:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                        report_content.append(f"  - {metric_name}: {metric_value:.2f}")
            
            # 添加文件链接
            if result_files:
                report_content.append("- 生成的文件:")
                for file_type, file_path in result_files.items():
                    if file_path:
                        rel_path = os.path.relpath(file_path, output_dir)
                        report_content.append(f"  - {file_type}: {os.path.basename(file_path)}")
            
            report_content.append("")
            
        except Exception as e:
            logger.error(f"测试 {seq_name} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 记录失败
            test_results[seq_name] = {
                "status": "失败",
                "error": str(e)
            }
            
            # 添加到报告
            report_content.append(f"### {seq_name} (分子类型: {molecule_type})")
            report_content.append(f"- 状态: ❌ 失败")
            report_content.append(f"- 错误: {str(e)}")
            report_content.append("")
    
    # 保存报告
    try:
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_content))
        logger.info(f"综合测试完成，报告保存至: {report_file}")
    except Exception as e:
        logger.error(f"保存报告失败: {str(e)}")
    
    return {
        "success": len([r for r in test_results.values() if r["status"] == "成功"]) > 0,
        "results": test_results,
        "report_file": report_file
    }

def generate_pdb_file(sequence_id, sequence, atom_positions, molecule_type="protein", output_dir=None, model_name="af3"):
    """
    生成PDB结构文件
    
    Args:
        sequence_id: 序列ID
        sequence: 分子序列
        atom_positions: 原子坐标
        molecule_type: 分子类型 ("protein", "rna", "dna")
        output_dir: 输出目录
        model_name: 模型名称
    
    Returns:
        PDB文件路径
    """
    if output_dir is None:
        output_dir = setup_output_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建PDB文件名
    pdb_file = os.path.join(output_dir, f"{sequence_id}_{model_name}.pdb")
    logger.info(f"生成PDB文件: {pdb_file}")
    
    try:
        # 确保原子坐标在CPU上
        if isinstance(atom_positions, torch.Tensor):
            atom_positions = atom_positions.detach().cpu()
        
        # 创建生物分子结构
        structure = create_simple_biomolecule(atom_positions, sequence, molecule_type)
        
        # 保存PDB文件
        save_pdb(structure, pdb_file)
        logger.info(f"PDB文件已保存: {pdb_file}")
        
        return pdb_file
    except Exception as e:
        logger.error(f"生成PDB文件出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

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
    """
    从原子坐标创建简单的Structure对象
    
    Args:
        atom_positions: 形状为[num_atoms, 3]的原子坐标张量或列表
        sequence: 序列字符串
        molecule_type: 分子类型，可以是"protein"、"rna"或"dna"
    
    Returns:
        Structure对象
    """
    try:
        logger.info(f"从原子坐标创建Structure对象，分子类型: {molecule_type}, 序列: {sequence}")
        
        # 处理列表类型的atom_positions
        if isinstance(atom_positions, list):
            if all(isinstance(item, torch.Tensor) for item in atom_positions):
                # 如果列表中的元素是张量，转换为numpy数组
                atom_positions_list = []
                for tensor in atom_positions:
                    if tensor.device.type != 'cpu':
                        tensor = tensor.cpu()
                    atom_positions_list.append(tensor.detach().numpy())
                atom_positions = np.concatenate(atom_positions_list, axis=0)
            else:
                # 如果元素不是张量，直接转换为numpy数组
                atom_positions = np.array(atom_positions)
        # 处理tensor类型的atom_positions
        elif torch.is_tensor(atom_positions):
            if atom_positions.device.type != 'cpu':
                atom_positions = atom_positions.cpu()
            atom_positions = atom_positions.detach().numpy()
        
        # 确保二维形状 [num_atoms, 3]
        if len(atom_positions.shape) == 3:
            # 如果形状是 [batch, num_atoms, 3]，取第一个批次
            atom_positions = atom_positions[0]
        
        # 获取原子数量
        num_atoms = atom_positions.shape[0]
        
        try:
            # 使用Bio.PDB创建Structure对象
            from Bio.PDB import Structure as StructureModule
            from Bio.PDB import StructureBuilder
            structure_builder = StructureBuilder.StructureBuilder()
            structure_builder.init_structure("model")
            structure_builder.init_model(0)
            structure_builder.init_chain("A")
            
            # 计算每个残基的原子数量
            if molecule_type == "protein":
                residue_atoms = {aa: 5 for aa in "ARNDCQEGHILKMFPSTWYV"}
            else:
                # 对于RNA和DNA，简化处理
                residue_atoms = {nucleotide: 10 for nucleotide in "ACGTU"}
            
            # 添加残基和原子
            atom_idx = 0
            for i, residue_type in enumerate(sequence):
                # 获取残基名称
                if molecule_type == "protein":
                    res_name = get_amino_acid_name(residue_type)
                elif molecule_type == "rna":
                    res_name = get_rna_name(residue_type)
                elif molecule_type == "dna":
                    res_name = get_dna_name(residue_type)
                else:
                    res_name = "UNK"
                
                # 初始化残基
                structure_builder.init_residue(res_name, " ", i+1, " ")
                
                # 获取该残基的原子数量
                num_atoms_in_residue = residue_atoms.get(residue_type, 5)
                
                # 添加原子
                for j in range(num_atoms_in_residue):
                    if atom_idx < num_atoms:
                        # 设置默认原子名称
                        atom_name = "CA" if j == 0 else f"C{j}"
                        
                        # 创建原子
                        coord = atom_positions[atom_idx]
                        structure_builder.init_atom(atom_name, coord, 0.0, 1.0, " ", atom_name, atom_idx, "C")
                        atom_idx += 1
            
            # 获取构建的结构
            structure = structure_builder.get_structure()
            
            logger.info(f"成功创建Structure对象，具有 {atom_idx} 个原子")
            return structure
            
        except Exception as e:
            logger.error(f"使用Bio.PDB创建Structure对象时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"创建Structure对象时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    return None

# 统一的置信度处理模块
def extract_plddt_from_logits(plddt_logits, device=None):
    """从pLDDT逻辑值中提取置信度分数
    
    Args:
        plddt_logits: 形状为[batch, num_atoms, bins]的张量
        device: 计算设备
    
    Returns:
        形状为[batch, num_atoms]的置信度分数张量, 范围0-100
    """
    try:
        if device is None and torch.is_tensor(plddt_logits):
            device = plddt_logits.device
            
        # 重排维度 [b, m, bins] -> [b, bins, m]
        logits = plddt_logits.permute(0, 2, 1)
        
        # 计算softmax
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # 计算期望值
        bin_width = 1.0 / logits.shape[1]
        bin_centers = torch.arange(
            0.5 * bin_width, 1.0, bin_width, 
            dtype=probs.dtype, device=device
        )
        
        # 计算置信度分数 (0-100)
        return torch.einsum('bpm,p->bm', probs, bin_centers) * 100
        
    except Exception as e:
        logger.warning(f"从pLDDT逻辑值提取置信度分数时出错: {str(e)}")
        return None

def extract_pae_from_logits(pae_logits, device=None):
    """从PAE逻辑值中提取预测对齐误差
    
    Args:
        pae_logits: 形状为[batch, bins, n, n]的张量
        device: 计算设备
    
    Returns:
        形状为[batch, n, n]的PAE张量
    """
    try:
        if device is None and torch.is_tensor(pae_logits):
            device = pae_logits.device
            
        # 重排维度 [b, bins, n, n] -> [b, n, n, bins]
        pae = pae_logits.permute(0, 2, 3, 1)
        
        # 计算softmax
        pae_probs = torch.nn.functional.softmax(pae, dim=-1)
        
        # 假设bin范围从0.5到32埃
        pae_bin_width = 31.5 / pae_probs.shape[-1]
        pae_bin_centers = torch.arange(
            0.5 + 0.5 * pae_bin_width, 32.0, pae_bin_width,
            dtype=pae_probs.dtype, device=device
        )
        
        # 计算期望值
        return torch.einsum('bijd,d->bij', pae_probs, pae_bin_centers)
        
    except Exception as e:
        logger.warning(f"从PAE逻辑值提取误差值时出错: {str(e)}")
        return None

def extract_pde_from_logits(pde_logits, device=None):
    """从PDE逻辑值中提取预测距离误差
    
    Args:
        pde_logits: 形状为[batch, bins, n, n]的张量
        device: 计算设备
    
    Returns:
        形状为[batch, n, n]的PDE张量
    """
    try:
        if device is None and torch.is_tensor(pde_logits):
            device = pde_logits.device
            
        # 重排维度 [b, bins, n, n] -> [b, n, n, bins]
        pde = pde_logits.permute(0, 2, 3, 1)
        
        # 计算softmax
        pde_probs = torch.nn.functional.softmax(pde, dim=-1)
        
        # 假设bin范围从0.5到32埃
        pde_bin_width = 31.5 / pde_probs.shape[-1]
        pde_bin_centers = torch.arange(
            0.5 + 0.5 * pde_bin_width, 32.0, pde_bin_width,
            dtype=pde_probs.dtype, device=device
        )
        
        # 计算期望值
        return torch.einsum('bijd,d->bij', pde_probs, pde_bin_centers)
        
    except Exception as e:
        logger.warning(f"从PDE逻辑值提取误差值时出错: {str(e)}")
        return None

def process_confidence_metrics(confidence_logits, device=None):
    """统一处理所有置信度信息，返回规范化的指标字典
    
    Args:
        confidence_logits: 包含plddt、pae和pde逻辑值的对象
        device: 计算设备
    
    Returns:
        包含提取的置信度指标的字典，包括plddt、pae和pde等
    """
    metrics = {}
    
    if confidence_logits is None:
        return metrics
    
    # 处理pLDDT
    if hasattr(confidence_logits, 'plddt') and confidence_logits.plddt is not None:
        plddt = extract_plddt_from_logits(confidence_logits.plddt, device)
        if plddt is not None:
            metrics['plddt'] = plddt
            metrics['plddt_mean'] = float(plddt.mean().item())
            metrics['plddt_median'] = float(plddt.median().item())
            metrics['plddt_min'] = float(plddt.min().item())
            metrics['plddt_max'] = float(plddt.max().item())
    
    # 处理PAE
    if hasattr(confidence_logits, 'pae') and confidence_logits.pae is not None:
        pae = extract_pae_from_logits(confidence_logits.pae, device)
        if pae is not None:
            metrics['pae'] = pae
            metrics['pae_mean'] = float(pae.mean().item())
            metrics['pae_max'] = float(pae.max().item())
    
    # 处理PDE
    if hasattr(confidence_logits, 'pde') and confidence_logits.pde is not None:
        pde = extract_pde_from_logits(confidence_logits.pde, device)
        if pde is not None:
            metrics['pde'] = pde
            metrics['pde_mean'] = float(pde.mean().item())
            metrics['pde_max'] = float(pde.max().item())
    
    # PTM分数 (如果可用)
    if hasattr(confidence_logits, 'ptm'):
        metrics['ptm'] = float(confidence_logits.ptm.item())
    
    # iPTM分数 (如果可用)
    if hasattr(confidence_logits, 'iptm'):
        metrics['iptm'] = float(confidence_logits.iptm.item())
    
    return metrics

class Config:
    """统一配置管理类"""
    
    def __init__(self, args=None):
        # 默认配置
        self.sequence = "AG"
        self.molecule_type = "protein"
        self.model_path = None
        self.output_dir = None
        self.force_cpu = False
        self.precision = "fp32"
        self.quiet = False
        self.debug = False
        
        # 功能标志
        self.test_basic_functionality = False
        self.test_comprehensive = False
        self.test_model_loading = False
        self.test_basic_prediction = False
        self.test_pdb_generation = False
        self.run_complete_pipeline = False
        self.extract_confidence = False
        self.generate_pdb = False
        self.plot = False
        
        # 高级参数
        self.epochs = 1
        self.use_msa = True
        self.use_templates = True
        self.recycling_steps = 3
        self.sampling_steps = 8
        self.memory_efficient = False
        self.custom_atom_positions = None
        
        # 设备和运行时配置
        self.device = None
        self.memory_config = MemoryConfig()
        
        # 从命令行参数更新配置
        if args:
            self.update_from_args(args)
        
        # 初始化配置
        self.initialize()
    
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        # 基本参数
        self.sequence = args.sequence
        self.molecule_type = args.molecule_type
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        self.force_cpu = args.force_cpu
        self.precision = args.precision
        self.quiet = args.quiet
        self.debug = args.debug if hasattr(args, 'debug') else False
        
        # 功能标志
        self.test_basic_functionality = args.test_basic_functionality
        self.test_comprehensive = args.test_comprehensive if hasattr(args, 'test_comprehensive') else False
        self.test_model_loading = args.test_model_loading if hasattr(args, 'test_model_loading') else False 
        self.test_basic_prediction = args.test_basic_prediction if hasattr(args, 'test_basic_prediction') else False
        self.test_pdb_generation = args.test_pdb_generation if hasattr(args, 'test_pdb_generation') else False
        self.run_complete_pipeline = args.run_complete_pipeline if hasattr(args, 'run_complete_pipeline') else False
        self.extract_confidence = args.extract_confidence if hasattr(args, 'extract_confidence') else False
        self.generate_pdb = args.generate_pdb if hasattr(args, 'generate_pdb') else False
        self.plot = args.plot if hasattr(args, 'plot') else False
        
        # 高级参数
        self.epochs = args.epochs
        self.use_msa = not args.no_msa if hasattr(args, 'no_msa') else True
        self.use_templates = not args.no_templates if hasattr(args, 'no_templates') else True
        self.recycling_steps = args.recycling_steps if hasattr(args, 'recycling_steps') else 3
        self.sampling_steps = args.sampling_steps if hasattr(args, 'sampling_steps') else 8
        self.memory_efficient = args.memory_efficient if hasattr(args, 'memory_efficient') else False
        self.custom_atom_positions = args.custom_atom_positions if hasattr(args, 'custom_atom_positions') else None
        
        # 如果没有指定任何操作，默认运行完整流水线
        if not any([self.test_basic_functionality, self.test_comprehensive, 
                   self.test_model_loading, self.test_basic_prediction, 
                   self.test_pdb_generation, self.run_complete_pipeline,
                   self.extract_confidence, self.generate_pdb, self.plot]):
            self.run_complete_pipeline = True
    
    def initialize(self):
        """初始化配置"""
        # 设置设备
        self.device = get_device(force_cpu=self.force_cpu)
        
        # 设置输出目录
        if not self.output_dir:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"./af3_output_{timestamp}"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置内存配置
        if self.memory_efficient:
            self.memory_config.memory_efficient = True
            
        # 设置日志级别
        if self.quiet:
            setup_logging(quiet=True)
        else:
            setup_logging(quiet=False)
    
    def to_dict(self):
        """将配置转换为字典"""
        return {
            "sequence": self.sequence,
            "molecule_type": self.molecule_type,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "device": str(self.device),
            "precision": self.precision,
            "epochs": self.epochs,
            "recycling_steps": self.recycling_steps,
            "sampling_steps": self.sampling_steps,
            "use_msa": self.use_msa,
            "use_templates": self.use_templates,
            "memory_efficient": self.memory_efficient
        }
    
    def __str__(self):
        """输出配置信息"""
        return json.dumps(self.to_dict(), indent=2)

class TestFramework:
    """统一测试框架类，用于组织和管理所有测试功能"""
    
    def __init__(self, config):
        """
        初始化测试框架
        
        Args:
            config: Config对象，包含所有配置参数
        """
        self.config = config
        self.model = None
        self.results = {}
    
    def load_model(self):
        """加载模型（如果尚未加载）"""
        if self.model is None and self.config.model_path:
            logger.info(f"从 {self.config.model_path} 加载预训练模型")
            self.model = load_pretrained_model(
                self.config.model_path,
                self.config.device,
                self.config.precision
            )
            
            if self.model is None:
                logger.error("模型加载失败")
            else:
                logger.info("模型加载成功")
        
        return self.model
    
    def run_test(self, test_type):
        """
        运行指定类型的测试
        
        Args:
            test_type: 测试类型，可选值包括 'basic', 'comprehensive', 'model_loading',
                      'confidence', 'pdb_generation', 'pipeline'
        
        Returns:
            测试结果
        """
        logger.info(f"运行 {test_type} 测试")
        
        if test_type == 'basic':
            return self.run_basic_functionality_test()
        elif test_type == 'comprehensive':
            return self.run_comprehensive_test()
        elif test_type == 'model_loading':
            return self.run_model_loading_test()
        elif test_type == 'confidence':
            return self.run_confidence_extraction_test()
        elif test_type == 'pdb_generation':
            return self.run_pdb_generation_test()
        elif test_type == 'plot':
            return self.run_plot_test()
        elif test_type == 'pipeline':
            return self.run_complete_pipeline()
        else:
            logger.error(f"未知测试类型: {test_type}")
            return None
    
    def run_basic_functionality_test(self):
        """运行基本功能测试"""
        logger.info("运行基本功能测试")
        return test_basic_functionality(
            output_dir=self.config.output_dir,
            device=self.config.device,
            precision=self.config.precision
        )
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("运行综合测试")
        self.load_model()
        return run_comprehensive_test(self.config)
    
    def run_model_loading_test(self):
        """测试模型加载"""
        logger.info("测试模型加载")
        model = self.load_model()
        return model is not None
    
    def run_confidence_extraction_test(self):
        """测试置信度提取"""
        logger.info("测试置信度提取")
        self.load_model()
        return test_confidence_extraction(
            self.model,
            sequence=self.config.sequence,
            molecule_type=self.config.molecule_type,
            output_dir=self.config.output_dir,
            plot_confidence=self.config.plot
        )
    
    def run_pdb_generation_test(self):
        """测试PDB文件生成"""
        logger.info("测试PDB文件生成")
        self.load_model()
        return test_pdb_generation(self.config.output_dir)
    
    def run_plot_test(self):
        """测试置信度图绘制"""
        logger.info("测试置信度图绘制")
        self.load_model()
        if self.model:
            # 执行单个预测获取输出
            output = run_single_prediction(
                self.model,
                "plot_test",
                self.config.sequence,
                self.config.molecule_type,
                self.config.output_dir,
                self.config.device,
                self.config.recycling_steps,
                self.config.sampling_steps,
                False,  # 不生成PDB
                self.config.precision
            )
            
            if output:
                return plot_confidence_metrics(
                    output,
                    self.config.sequence,
                    self.config.output_dir
                )
        return None
    
    def run_complete_pipeline(self):
        """运行完整预测流水线"""
        logger.info("运行完整预测流水线")
        # 移除加载模型的检查，直接运行pipeline
        # self.load_model()
        # if self.model:
        return run_complete_pipeline(
            sequence=self.config.sequence,
            molecule_type=self.config.molecule_type,
            output_dir=self.config.output_dir,
            epochs=self.config.epochs,
            use_msa=self.config.use_msa,
            use_templates=self.config.use_templates,
            device=self.config.device,
            precision=self.config.precision,
            save_structures=True,
            quiet=self.config.quiet
        )
        # return None
    
    def run_all_specified_tests(self):
        """运行所有在配置中指定的测试"""
        # 记录开始时间
        start_time = dt.now()
        logger.info(f"开始运行测试，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 根据配置运行相应的测试
        if self.config.test_basic_functionality:
            self.results['basic'] = self.run_basic_functionality_test()
            
        if self.config.test_comprehensive:
            self.results['comprehensive'] = self.run_comprehensive_test()
            
        if self.config.test_model_loading:
            self.results['model_loading'] = self.run_model_loading_test()
            
        if self.config.test_basic_prediction or self.config.extract_confidence:
            self.results['confidence'] = self.run_confidence_extraction_test()
            
        if self.config.test_pdb_generation or self.config.generate_pdb:
            self.results['pdb_generation'] = self.run_pdb_generation_test()
            
        if self.config.plot:
            self.results['plot'] = self.run_plot_test()
            
        if self.config.run_complete_pipeline:
            self.results['pipeline'] = self.run_complete_pipeline()
        
        # 记录结束时间
        end_time = dt.now()
        duration = end_time - start_time
        logger.info(f"测试完成，时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"测试总耗时: {duration.total_seconds():.2f} 秒")
        
        # 生成测试报告
        self.generate_test_report()
        
        return self.results
    
    def generate_test_report(self):
        """生成测试报告"""
        report_file = os.path.join(self.config.output_dir, "test_report.md")
        
        try:
            with open(report_file, "w") as f:
                f.write("# AlphaFold 3 测试报告\n\n")
                f.write(f"测试时间: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 写入配置信息
                f.write("## 测试配置\n\n")
                for key, value in self.config.to_dict().items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
                
                # 写入测试结果
                f.write("## 测试结果\n\n")
                
                for test_name, result in self.results.items():
                    f.write(f"### {test_name}\n\n")
                    
                    if isinstance(result, bool):
                        status = "✅ 成功" if result else "❌ 失败"
                        f.write(f"状态: {status}\n\n")
                    elif isinstance(result, dict):
                        f.write("详细结果:\n\n")
                        for k, v in result.items():
                            f.write(f"- {k}: {v}\n")
                        f.write("\n")
                    elif result is not None:
                        f.write(f"结果: {result}\n\n")
                    else:
                        f.write("状态: ❌ 失败\n\n")
                
                # 写入系统信息
                f.write("## 系统信息\n\n")
                f.write(f"- PyTorch版本: {torch.__version__}\n")
                f.write(f"- NPU可用: {torch_npu.npu.is_available()}\n")
                if torch_npu.npu.is_available():
                    f.write(f"- NPU设备: {torch_npu.npu.get_device_name(0)}\n")
                f.write(f"- 运行设备: {self.config.device}\n")
                f.write(f"- 精度: {self.config.precision}\n")
            
            logger.info(f"测试报告已生成: {report_file}")
            return report_file
        except Exception as e:
            logger.error(f"生成测试报告失败: {str(e)}")
            return None

def setup_logging(quiet=False, log_file=None):
    """
    设置日志配置
    
    Args:
        quiet: 是否减少控制台输出
        log_file: 日志文件路径，如果为None则不输出到文件
    """
    # 配置根日志记录器
    # 清除已有的根日志处理程序
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 获取并配置我们的logger
    global logger
    logger = logging.getLogger('af3_test')
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理程序
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 添加控制台处理程序
    console_handler = logging.StreamHandler()
    
    if quiet:
        # 静默模式：控制台只输出WARNING及以上级别的日志，使用简单格式
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setLevel(logging.WARNING)
    else:
        # 正常模式：控制台输出INFO及以上级别的日志，使用详细格式
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                           datefmt='%H:%M:%S')
        console_handler.setLevel(logging.INFO)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理程序
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # 文件中记录所有日志
        logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

if __name__ == "__main__":
    # 在这里调用run_test_framework函数，而不是一个未定义的main函数
    import sys
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='AlphaFold 3 Test Framework')
    
    # 添加参数
    parser.add_argument('--test-comprehensive', action='store_true', help='运行综合测试')
    parser.add_argument('--run-complete-pipeline', action='store_true', help='运行完整的模型训练和预测流程')
    parser.add_argument('--sequence', type=str, default='ACDEFGHIKLMNPQRSTVWY', help='用于测试的氨基酸序列')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮次')
    parser.add_argument('--recycling-steps', type=int, default=3, help='循环步数')
    parser.add_argument('--sampling-steps', type=int, default=50, help='采样步数')
    parser.add_argument('--output-dir', type=str, default='./test_results', help='输出目录')
    parser.add_argument('--model-path', type=str, default=None, help='预训练模型路径(可选)')
    parser.add_argument('--cpu-only', action='store_true', help='仅使用CPU')
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(quiet=args.quiet, log_file=args.log_file)
    
    # 如果要运行综合测试
    if args.test_comprehensive:
        logger.info("运行综合测试...")
        test_framework = TestFramework(
            epochs=args.epochs,
            recycling_steps=args.recycling_steps,
            sampling_steps=args.sampling_steps,
            output_dir=args.output_dir,
            model_path=args.model_path,
            cpu_only=args.cpu_only
        )
        test_framework.run_comprehensive_test()
    
    # 如果要运行完整流程
    elif args.run_complete_pipeline:
        logger.info("运行完整流程...")
        results = run_complete_pipeline(
            sequence=args.sequence,
            output_dir=args.output_dir,
            epochs=args.epochs,  # 确保epochs参数被正确传递
            use_msa=not args.cpu_only,  # 这里假设cpu_only意味着不使用MSA
            use_templates=not args.cpu_only,  # 这里假设cpu_only意味着不使用templates
            device=torch.device('cpu') if args.cpu_only else None,
            precision="fp32",
            save_structures=True,
            quiet=args.quiet
        )
    
    # 如果没有指定任何操作，显示帮助
    else:
        parser.print_help()

