#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaFold 3 蛋白质结构预测测试脚本
系统地测试不同复杂度的蛋白质序列，从简单到复杂
"""

import os
import time
import argparse
import json
import tempfile
from datetime import datetime
import torch
import numpy as np
import traceback
import gc
import sys
from run_af3_pt3 import run_alphafold3, get_atom_counts_per_residue

import torch_npu

# 测试序列集合
TEST_SEQUENCES = {
    # 简单序列 - 与 run_af3_pt2.py 相同的序列
    "AG": "AG",  # 丙氨酸-甘氨酸二肽，2个氨基酸
    
    # 短肽序列
    "RGD": "RGD",  # 精氨酸-甘氨酸-天冬氨酸，细胞粘附序列，3个氨基酸
    "GGPP": "GGPP",  # 甘氨酸-甘氨酸-脯氨酸-脯氨酸，4个氨基酸
    "YGGFL": "YGGFL",  # 亮氨酸脑啡肽，5个氨基酸
    
    # 已知结构的小型蛋白质
    "insulin_a": "GIVEQCCTSICSLYQLENYCN",  # 胰岛素A链，21个氨基酸
    "insulin_b": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # 胰岛素B链，30个氨基酸
    "melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",  # 蜜蜂毒素，26个氨基酸
    
    # 中等大小的蛋白质
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 泛素，76个氨基酸
    "cytochrome_c": "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",  # 细胞色素C，104个氨基酸
    
    # 较大的蛋白质
    "lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",  # 溶菌酶，129个氨基酸
    
    # 未知结构的蛋白质序列（合成序列）
    "synthetic_1": "MAAGTKLVLVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS",  # 合成序列1，约188个氨基酸
}

# 测试配置
TEST_CONFIGS = {
    "quick": ["AG", "RGD", "YGGFL"],  # 快速测试，只测试简单序列
    "small": ["AG", "RGD", "YGGFL", "insulin_a", "insulin_b", "melittin"],  # 小型测试，包括简单序列和小型蛋白质
    "medium": ["AG", "insulin_b", "melittin", "ubiquitin"],  # 中等测试，包括代表性序列
    "full": list(TEST_SEQUENCES.keys()),  # 完整测试，测试所有序列
    "custom": []  # 自定义测试，由用户指定
}

# 内存优化配置
class MemoryConfig:
    def __init__(self, memory_efficient=False):
        self.memory_efficient = memory_efficient
        self.auto_cpu_fallback = True
        self.clear_cache_between_tests = True
        self.gradient_checkpointing = True

# TODO 把这个迁移成NPU用的
    def optimize_for_cuda(self):
        """优化NPU内存使用"""
        if not torch_npu.npu.is_available():
            return
        
        if self.memory_efficient:
            print("启用内存效率模式，减少GPU内存使用")
            
            # 设置自动调整内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # 清空缓存
            self.clear_cuda_cache()
            
            # 设置默认张量类型为CPU
            torch.set_default_dtype(torch.float32)

    def clear_cuda_cache(self):
        """清空CUDA缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def setup_output_dir(base_dir=None):
    """设置输出目录，如果未指定则使用系统临时目录"""
    if base_dir is None:
        # 使用系统临时目录
        base_dir = os.path.join(tempfile.gettempdir(), "af3_test_results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"test_{timestamp}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        # 如果没有权限，则使用临时目录
        temp_dir = os.path.join(tempfile.gettempdir(), "af3_test_results")
        print(f"警告: 无法创建目录 '{output_dir}'，将使用临时目录: '{temp_dir}'")
        os.makedirs(temp_dir, exist_ok=True)
        output_dir = os.path.join(temp_dir, f"test_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def calculate_total_atoms(sequence):
    """计算序列中的总原子数"""
    atom_counts = get_atom_counts_per_residue()
    total_atoms = 0
    for aa in sequence:
        total_atoms += atom_counts.get(aa, 5)
    return total_atoms

def run_test(sequence_name, sequence, output_dir, train_steps=50, device=None, verbose=True, memory_config=None):
    """运行单个序列的测试"""
    if memory_config is None:
        memory_config = MemoryConfig()
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"测试序列: {sequence_name}")
        print(f"序列: {sequence}")
        print(f"长度: {len(sequence)}个氨基酸")
        print(f"预计原子数: {calculate_total_atoms(sequence)}")
        print(f"训练步骤: {train_steps}")
        print(f"{'='*50}")
    
    # 创建序列特定的输出目录
    seq_output_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(seq_output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 根据序列长度动态调整训练步骤
    actual_train_steps = train_steps
    if len(sequence) > 30:
        # 较长序列使用更少的训练步骤
        actual_train_steps = max(5, int(train_steps / (len(sequence) / 30)))
        if verbose:
            print(f"序列较长，调整训练步骤为 {actual_train_steps}")
    
    # 优化CUDA内存使用
    if memory_config.clear_cache_between_tests:
        memory_config.clear_cuda_cache()
    
    # 运行 AlphaFold 3
    try:
        original_device = device
        
        # 运行预测
        result = run_alphafold3(sequence, seq_output_dir, actual_train_steps, device)
        
        # 处理可能的元组返回值
        if isinstance(result, tuple) and len(result) == 2:
            sampled_atom_pos, confidence_score = result
            has_confidence = True
            if verbose:
                print("获取到预测结构和置信度信息")
        else:
            sampled_atom_pos = result
            confidence_score = None
            has_confidence = False
            if verbose:
                print("仅获取到预测结构，无置信度信息")
        
        success = sampled_atom_pos is not None
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 记录结果
        test_result = {
            "sequence_name": sequence_name,
            "sequence": sequence,
            "length": len(sequence),
            "expected_atoms": calculate_total_atoms(sequence),
            "success": success,
            "elapsed_time": elapsed_time,
            "train_steps": actual_train_steps,
            "device": str(device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence_scores": {}  # 添加置信度分数的字典
        }
        
        if sampled_atom_pos is not None:
            test_result["result_shape"] = list(map(int, sampled_atom_pos.shape))
            test_result["mean_value"] = float(sampled_atom_pos.mean().item())
            test_result["std_value"] = float(sampled_atom_pos.std().item())
            
            # 如果有置信度信息，添加到结果
            if has_confidence and confidence_score is not None:
                try:
                    # 添加pLDDT分数
                    if hasattr(confidence_score, 'plddt'):
                        plddt = confidence_score.plddt
                        
                        # 添加到结果
                        test_result["confidence_scores"]["mean_plddt"] = float(plddt.mean().item())
                        test_result["confidence_scores"]["min_plddt"] = float(plddt.min().item())
                        test_result["confidence_scores"]["max_plddt"] = float(plddt.max().item())
                        
                        # 计算每个残基的平均pLDDT分数
                        residue_plddt = []
                        atom_idx = 0
                        atom_counts = get_atom_counts_per_residue()
                        
                        for aa in sequence:
                            num_atoms = atom_counts.get(aa, 5)
                            if atom_idx + num_atoms <= plddt.shape[1]:
                                residue_score = plddt[0, atom_idx:atom_idx+num_atoms].mean().item()
                                residue_plddt.append(float(residue_score))
                            atom_idx += num_atoms
                        
                        # 添加残基级别的置信度
                        test_result["confidence_scores"]["residue_plddt"] = residue_plddt
                        
                        if verbose:
                            print(f"\n置信度信息:")
                            print(f"  平均pLDDT分数: {plddt.mean().item():.4f}")
                            print(f"  最小pLDDT分数: {plddt.min().item():.4f}")
                            print(f"  最大pLDDT分数: {plddt.max().item():.4f}")
                            
                            if residue_plddt:
                                print("\n残基级别的置信度分数:")
                                for i, (aa, score) in enumerate(zip(sequence, residue_plddt)):
                                    quality = "很高" if score > 90 else "高" if score > 70 else "中等" if score > 50 else "低"
                                    print(f"  残基 {i+1} ({aa}): {score:.4f} - 质量{quality}")
                    
                    # 如果存在pTM分数
                    if hasattr(confidence_score, 'ptm'):
                        ptm = confidence_score.ptm
                        test_result["confidence_scores"]["ptm"] = float(ptm.item())
                        
                        if verbose:
                            print(f"  pTM分数: {ptm.item():.4f}")
                    
                    # 如果存在接口pTM分数
                    if hasattr(confidence_score, 'iptm') and confidence_score.iptm is not None:
                        iptm = confidence_score.iptm
                        test_result["confidence_scores"]["iptm"] = float(iptm.item())
                        
                        if verbose:
                            print(f"  接口pTM分数: {iptm.item():.4f}")
                except Exception as ce:
                    if verbose:
                        print(f"处理置信度信息时出错: {str(ce)}")
        
        # 保存结果到JSON文件
        result_file = os.path.join(seq_output_dir, "test_result.json")
        with open(result_file, 'w') as f:
            json.dump(test_result, f, indent=2)
        
        if verbose:
            print(f"\n测试结果:")
            print(f"  成功: {'是' if success else '否'}")
            print(f"  耗时: {elapsed_time:.2f}秒")
            if sampled_atom_pos is not None:
                print(f"  结果形状: {sampled_atom_pos.shape}")
                print(f"  平均值: {sampled_atom_pos.mean().item():.4f}")
                print(f"  标准差: {sampled_atom_pos.std().item():.4f}")
        
        return test_result
    
    except RuntimeError as e:
        # 检查是否为CUDA内存错误
        if "CUDA out of memory" in str(e) and memory_config.auto_cpu_fallback and device.type == 'cuda':
            if verbose:
                print(f"\nCUDA内存不足，尝试使用CPU进行计算...")
            
            # 清理内存
            memory_config.clear_cuda_cache()
            
            # 尝试使用CPU重新运行
            device = torch.device('cpu')
            
            try:
                # 减少训练步骤，避免CPU计算过慢
                fallback_train_steps = min(10, actual_train_steps)
                if verbose:
                    print(f"使用CPU运行，减少训练步骤至 {fallback_train_steps}")
                
                result = run_alphafold3(sequence, seq_output_dir, fallback_train_steps, device)
                success = result is not None
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # 记录结果
                test_result = {
                    "sequence_name": sequence_name,
                    "sequence": sequence,
                    "length": len(sequence),
                    "expected_atoms": calculate_total_atoms(sequence),
                    "success": success,
                    "elapsed_time": elapsed_time,
                    "train_steps": fallback_train_steps,
                    "device": "cpu (fallback from cuda)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if result is not None:
                    test_result["result_shape"] = list(map(int, result.shape))
                    test_result["mean_value"] = float(result.mean().item())
                    test_result["std_value"] = float(result.std().item())
                
                # 保存结果到JSON文件
                result_file = os.path.join(seq_output_dir, "test_result.json")
                with open(result_file, 'w') as f:
                    json.dump(test_result, f, indent=2)
                
                if verbose:
                    print(f"\n测试结果 (CPU回退):")
                    print(f"  成功: {'是' if success else '否'}")
                    print(f"  耗时: {elapsed_time:.2f}秒")
                    if result is not None:
                        print(f"  结果形状: {result.shape}")
                        print(f"  平均值: {result.mean().item():.4f}")
                        print(f"  标准差: {result.std().item():.4f}")
                
                return test_result
            
            except Exception as fallback_error:
                if verbose:
                    print(f"CPU回退也失败: {str(fallback_error)}")
                # 继续处理原始错误
                pass
        
        # 通用错误处理
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 记录错误
        test_result = {
            "sequence_name": sequence_name,
            "sequence": sequence,
            "length": len(sequence),
            "expected_atoms": calculate_total_atoms(sequence),
            "success": False,
            "elapsed_time": elapsed_time,
            "train_steps": actual_train_steps,
            "device": str(device),
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果到JSON文件
        result_file = os.path.join(seq_output_dir, "test_result.json")
        with open(result_file, 'w') as f:
            json.dump(test_result, f, indent=2)
        
        if verbose:
            print(f"\n测试失败:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误: {str(e)}")
            print(f"  耗时: {elapsed_time:.2f}秒")
            print(f"  详细错误信息已保存至: {result_file}")
        
        return test_result
    
    except Exception as e:
        # 处理其他异常
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 记录错误
        test_result = {
            "sequence_name": sequence_name,
            "sequence": sequence,
            "length": len(sequence),
            "expected_atoms": calculate_total_atoms(sequence),
            "success": False,
            "elapsed_time": elapsed_time,
            "train_steps": actual_train_steps,
            "device": str(device),
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果到JSON文件
        result_file = os.path.join(seq_output_dir, "test_result.json")
        with open(result_file, 'w') as f:
            json.dump(test_result, f, indent=2)
        
        if verbose:
            print(f"\n测试失败:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误: {str(e)}")
            print(f"  耗时: {elapsed_time:.2f}秒")
            print(f"  详细错误信息已保存至: {result_file}")
        
        return test_result

def run_test_suite(test_config, output_dir, train_steps=50, device=None, verbose=True, memory_config=None):
    """运行测试套件"""
    if memory_config is None:
        memory_config = MemoryConfig()
    
    # 确定要测试的序列
    sequences_to_test = []
    if isinstance(test_config, list):
        # 如果提供了序列名称列表
        for seq_name in test_config:
            if seq_name in TEST_SEQUENCES:
                sequences_to_test.append((seq_name, TEST_SEQUENCES[seq_name]))
            else:
                print(f"警告: 未知的序列名称 '{seq_name}'，将被跳过")
    elif test_config in TEST_CONFIGS:
        # 如果提供了预定义的测试配置
        for seq_name in TEST_CONFIGS[test_config]:
            sequences_to_test.append((seq_name, TEST_SEQUENCES[seq_name]))
    else:
        # 如果提供了自定义序列
        sequences_to_test.append(("custom", test_config))
    
    # 创建测试套件结果目录
    suite_output_dir = os.path.join(output_dir, "test_suite")
    os.makedirs(suite_output_dir, exist_ok=True)
    
    # 运行测试
    results = []
    successful_tests = 0
    failed_tests = 0
    
    for i, (seq_name, sequence) in enumerate(sequences_to_test):
        if verbose:
            print(f"\n[{i+1}/{len(sequences_to_test)}] 测试 {seq_name}...")
        
        result = run_test(seq_name, sequence, suite_output_dir, train_steps, device, verbose, memory_config)
        results.append(result)
        
        if result["success"]:
            successful_tests += 1
        else:
            failed_tests += 1
        
        # 在测试之间清理内存
        if memory_config.clear_cache_between_tests:
            memory_config.clear_cuda_cache()
        
        # 显示进度
        if verbose:
            print(f"\n进度: {i+1}/{len(sequences_to_test)} 完成 ({successful_tests}成功/{failed_tests}失败)")
    
    # 汇总结果
    summary = {
        "total_tests": len(results),
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "total_time": sum(r["elapsed_time"] for r in results),
        "results": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_summary": {}
    }
    
    # 添加置信度汇总信息
    tests_with_confidence = 0
    total_plddt = 0.0
    min_plddt = float('inf')
    max_plddt = float('-inf')
    
    # 处理每个测试结果的置信度信息
    for result in results:
        if result["success"] and "confidence_scores" in result and result["confidence_scores"]:
            conf_scores = result["confidence_scores"]
            if "mean_plddt" in conf_scores:
                tests_with_confidence += 1
                total_plddt += conf_scores["mean_plddt"]
                
                if "min_plddt" in conf_scores and conf_scores["min_plddt"] < min_plddt:
                    min_plddt = conf_scores["min_plddt"]
                
                if "max_plddt" in conf_scores and conf_scores["max_plddt"] > max_plddt:
                    max_plddt = conf_scores["max_plddt"]
    
    # 如果有置信度信息，添加到汇总中
    if tests_with_confidence > 0:
        summary["confidence_summary"] = {
            "tests_with_confidence": tests_with_confidence,
            "average_plddt": total_plddt / tests_with_confidence,
        }
        
        if min_plddt != float('inf'):
            summary["confidence_summary"]["min_plddt"] = min_plddt
        
        if max_plddt != float('-inf'):
            summary["confidence_summary"]["max_plddt"] = max_plddt
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "test_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"测试套件完成")
        print(f"总测试数: {summary['total_tests']}")
        print(f"成功测试: {summary['successful_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"总耗时: {summary['total_time']:.2f}秒")
        
        # 打印置信度汇总信息
        if tests_with_confidence > 0:
            print(f"\n置信度汇总信息:")
            print(f"  包含置信度的测试: {tests_with_confidence}/{summary['total_tests']}")
            print(f"  平均pLDDT分数: {summary['confidence_summary']['average_plddt']:.4f}")
            
            if "min_plddt" in summary["confidence_summary"]:
                print(f"  最小pLDDT分数: {summary['confidence_summary']['min_plddt']:.4f}")
            
            if "max_plddt" in summary["confidence_summary"]:
                print(f"  最大pLDDT分数: {summary['confidence_summary']['max_plddt']:.4f}")
        
        print(f"结果保存在: {summary_file}")
        print(f"{'='*50}")
    
    return summary

def save_confidence_to_csv(output_dir, results):
    """将置信度信息保存到CSV文件中，便于进一步分析"""
    import csv
    
    confidence_file = os.path.join(output_dir, "confidence_scores.csv")
    
    # 准备CSV文件列
    header = ["sequence_name", "sequence_length", "mean_plddt"]
    
    # 检查是否有残基级别的置信度
    has_residue_scores = False
    max_residue_count = 0
    
    for result in results:
        if (result.get("success", False) and 
            "confidence_scores" in result and 
            "residue_plddt" in result["confidence_scores"]):
            
            has_residue_scores = True
            residue_count = len(result["confidence_scores"]["residue_plddt"])
            max_residue_count = max(max_residue_count, residue_count)
    
    # 如果有残基级别的置信度，添加残基列
    if has_residue_scores:
        for i in range(max_residue_count):
            header.append(f"residue_{i+1}")
    
    # 写入CSV文件
    with open(confidence_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for result in results:
            if not result.get("success", False) or "confidence_scores" not in result:
                continue
            
            conf_scores = result["confidence_scores"]
            if "mean_plddt" not in conf_scores:
                continue
            
            row = [
                result["sequence_name"],
                result["length"],
                conf_scores["mean_plddt"]
            ]
            
            # 添加残基级别的置信度
            if has_residue_scores and "residue_plddt" in conf_scores:
                residue_scores = conf_scores["residue_plddt"]
                for score in residue_scores:
                    row.append(score)
                
                # 填充缺失的残基
                while len(row) < len(header):
                    row.append("")
            
            writer.writerow(row)
    
    return confidence_file

def plot_confidence_scores(output_dir, results):
    """绘制置信度分数图表，用于可视化分析预测质量"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端，适用于无GUI环境
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("无法绘制图表，请安装matplotlib: pip install matplotlib")
        return None
    
    # 检查是否有有效的置信度结果
    valid_results = []
    for result in results:
        if (result.get("success", False) and 
            "confidence_scores" in result and 
            "residue_plddt" in result["confidence_scores"] and
            len(result["confidence_scores"]["residue_plddt"]) > 0):
            valid_results.append(result)
    
    if not valid_results:
        print("没有足够的置信度数据进行绘图")
        return None
    
    # 设置图表
    plt.figure(figsize=(12, 8))
    
    # 绘制平均pLDDT分数条形图
    ax1 = plt.subplot(211)
    sequence_names = []
    mean_scores = []
    
    for result in valid_results:
        sequence_names.append(result["sequence_name"])
        mean_scores.append(result["confidence_scores"]["mean_plddt"])
    
    # 设置颜色，基于pLDDT分数范围
    colors = []
    for score in mean_scores:
        if score >= 90:
            colors.append('darkblue')
        elif score >= 70:
            colors.append('blue')
        elif score >= 50:
            colors.append('lightblue')
        else:
            colors.append('gray')
    
    ax1.bar(sequence_names, mean_scores, color=colors)
    ax1.set_ylabel('平均pLDDT分数')
    ax1.set_title('各序列的平均pLDDT置信度分数')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.5)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # 添加分数质量标记
    ax1.text(len(sequence_names), 95, '很高', color='darkblue', horizontalalignment='right')
    ax1.text(len(sequence_names), 75, '高', color='blue', horizontalalignment='right')
    ax1.text(len(sequence_names), 55, '中等', color='lightblue', horizontalalignment='right')
    ax1.text(len(sequence_names), 35, '低', color='gray', horizontalalignment='right')
    
    plt.xticks(rotation=45, ha='right')
    
    # 绘制残基级别pLDDT分数热图（只取前10个序列，避免图表过大）
    max_sequences = min(10, len(valid_results))
    max_residues = max(len(result["confidence_scores"]["residue_plddt"]) for result in valid_results[:max_sequences])
    
    if max_residues > 1:  # 只有当有多个残基时才绘制热图
        ax2 = plt.subplot(212)
        
        # 准备热图数据
        heatmap_data = np.zeros((max_sequences, max_residues))
        heatmap_labels = []
        
        for i, result in enumerate(valid_results[:max_sequences]):
            residue_scores = result["confidence_scores"]["residue_plddt"]
            heatmap_data[i, :len(residue_scores)] = residue_scores
            heatmap_labels.append(f"{result['sequence_name']} ({result['length']}aa)")
        
        # 绘制热图
        cmap = plt.cm.get_cmap('viridis')
        im = ax2.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=100)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('pLDDT分数')
        
        # 设置标签
        ax2.set_yticks(np.arange(len(heatmap_labels)))
        ax2.set_yticklabels(heatmap_labels)
        ax2.set_title('残基级别pLDDT置信度分数')
        ax2.set_xlabel('残基位置')
    
    # 调整布局并保存
    plt.tight_layout()
    confidence_plot_file = os.path.join(output_dir, "confidence_plot.png")
    plt.savefig(confidence_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return confidence_plot_file

def main():
    parser = argparse.ArgumentParser(description="AlphaFold 3 蛋白质结构预测测试脚本")
    parser.add_argument("-c", "--config", choices=list(TEST_CONFIGS.keys()), default="quick",
                        help=f"测试配置: {', '.join(TEST_CONFIGS.keys())} (默认: quick)")
    parser.add_argument("-s", "--sequence", help="自定义蛋白质序列")
    parser.add_argument("-n", "--name", help="自定义序列名称 (与 -s 一起使用)")
    parser.add_argument("-l", "--list", action="store_true", help="列出所有可用的测试序列")
    parser.add_argument("-o", "--output", help="输出目录 (默认: 系统临时目录)")
    parser.add_argument("-t", "--train_steps", type=int, default=50, help="每个序列的训练步骤数 (默认: 50)")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细输出")
    parser.add_argument("--memory-efficient", action="store_true", help="内存效率模式，适用于GPU内存有限的情况")
    parser.add_argument("--no-auto-fallback", action="store_true", help="禁用自动CPU回退")
    parser.add_argument("--single-test", action="store_true", help="测试单个序列并退出")
    parser.add_argument("--save-confidence-csv", action="store_true", help="将置信度信息保存到CSV文件中")
    parser.add_argument("--plot-confidence", action="store_true", help="绘制置信度分数图表")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 内存优化
    memory_config = MemoryConfig(args.memory_efficient)
    if args.no_auto_fallback:
        memory_config.auto_cpu_fallback = False
    
    memory_config.optimize_for_cuda()
    
    # 列出所有可用的测试序列
    if args.list:
        print("\n可用的测试序列:")
        for config_name, seq_names in TEST_CONFIGS.items():
            if config_name != "custom":
                print(f"\n{config_name} 配置:")
                for seq_name in seq_names:
                    sequence = TEST_SEQUENCES[seq_name]
                    print(f"  {seq_name}: {len(sequence)}个氨基酸, {calculate_total_atoms(sequence)}个原子")
                    if len(sequence) <= 30:
                        print(f"    序列: {sequence}")
                    else:
                        print(f"    序列: {sequence[:15]}...{sequence[-15:]}")
        return
    
    # 设置输出目录
    output_dir = setup_output_dir(args.output)
    print(f"输出目录: {output_dir}")
    
    # 优化训练步骤
    train_steps = args.train_steps
    
    # 确定测试内容
    results = []
    
    if args.sequence:
        # 使用自定义序列
        seq_name = args.name if args.name else "custom"
        print(f"使用自定义序列: {seq_name}")
        result = run_test(seq_name, args.sequence, output_dir, train_steps, device, True, memory_config)
        results.append(result)
    elif args.single_test:
        # 只测试AG序列
        print("运行单个简单测试: AG")
        result = run_test("AG", TEST_SEQUENCES["AG"], output_dir, train_steps, device, True, memory_config)
        results.append(result)
    else:
        # 使用预定义的测试配置
        print(f"使用测试配置: {args.config}")
        summary = run_test_suite(args.config, output_dir, train_steps, device, True, memory_config)
        results = summary.get("results", [])
    
    # 可选：保存置信度到CSV
    if args.save_confidence_csv and results:
        try:
            confidence_csv = save_confidence_to_csv(output_dir, results)
            print(f"\n置信度数据已保存到: {confidence_csv}")
        except Exception as csv_error:
            print(f"保存置信度CSV文件时出错: {str(csv_error)}")
    
    # 可选：绘制置信度图表
    if args.plot_confidence and results:
        try:
            confidence_plot = plot_confidence_scores(output_dir, results)
            if confidence_plot:
                print(f"置信度图表已保存到: {confidence_plot}")
        except Exception as plot_error:
            print(f"绘制置信度图表时出错: {str(plot_error)}")
            print(traceback.format_exc())

# 当以脚本形式运行时执行main函数
if __name__ == "__main__":
    main()