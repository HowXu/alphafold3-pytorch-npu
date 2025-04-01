import torch
import torch_npu  # 添加华为NPU支持
import time
import numpy as np
from operator_deliver.cdist.cdist_impl import custom_cdist
import json
from datetime import datetime
import os

class TestResultLogger:
    def __init__(self, output_dir='test_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {
            'timestamp': self.timestamp,
            'accuracy_tests': [],
            'performance_tests': [],
            'scipy_equivalence': []
        }
    
    def log_accuracy_test(self, p, dtype, shape, compute_mode, test_result):
        self.results['accuracy_tests'].append({
            'p': str(p),
            'dtype': str(dtype),
            'shape': shape,
            'compute_mode': compute_mode,
            'test_result': test_result
        })
    
    def log_performance_test(self, p, shape, compute_mode, perf_result):
        self.results['performance_tests'].append({
            'p': str(p),
            'shape': shape,
            'compute_mode': compute_mode,
            'performance': perf_result
        })
    
    def log_scipy_test(self, p, test_result):
        self.results['scipy_equivalence'].append({
            'p': str(p),
            'test_result': test_result
        })
    
    def save_results(self):
        filename = f'cdist_test_results_{self.timestamp}.json'
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 生成可读性更好的文本报告
        report_file = os.path.join(self.output_dir, f'cdist_test_report_{self.timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("cdist算子测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("准确度测试结果:\n")
            for test in self.results['accuracy_tests']:
                f.write(f"\np={test['p']}, dtype={test['dtype']}\n")
                f.write(f"形状: batch_size={test['shape'][0]}, p1={test['shape'][1]}, "
                       f"p2={test['shape'][2]}, m={test['shape'][3]}\n")
                if 'compute_mode' in test:
                    f.write(f"计算模式: {test['compute_mode']}\n")
                f.write(f"测试结果:\n")
                for k, v in test['test_result'].items():
                    f.write(f"  {k}: {v}\n")
            
            f.write("\n性能测试结果:\n")
            for test in self.results['performance_tests']:
                f.write(f"\np={test['p']}\n")
                f.write(f"形状: batch_size={test['shape'][0]}, p1={test['shape'][1]}, "
                       f"p2={test['shape'][2]}, m={test['shape'][3]}\n")
                if 'compute_mode' in test:
                    f.write(f"计算模式: {test['compute_mode']}\n")
                f.write("性能结果:\n")
                for k, v in test['performance'].items():
                    f.write(f"  {k}: {v}\n")
            
            f.write("\nSciPy等价性测试结果:\n")
            for test in self.results['scipy_equivalence']:
                f.write(f"\np={test['p']}\n")
                f.write(f"测试结果: {'通过' if test['test_result']['is_equivalent'] else '失败'}\n")
                if 'error_details' in test['test_result']:
                    f.write(f"错误详情: {test['test_result']['error_details']}\n")

def test_cdist_accuracy(logger, device_cpu='cpu', device_npu='npu', p=2.0, rtol=1e-5, atol=1e-5):
    """
    测试自定义cdist算子的准确度
    
    参数:
        device_cpu: CPU设备
        device_npu: NPU设备
        p: p范数的p值
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
    """
    print(f"\n测试cdist算子准确度 (p={p}):")
    
    # 测试不同数据类型
    dtypes = [torch.float32]
    if p != 0:  # 汉明距离只支持float32
        dtypes.extend([torch.float16, torch.bfloat16])
    
    for dtype in dtypes:
        print(f"\n测试数据类型: {dtype}")
        
        # 测试不同规模的输入
        test_shapes = [
            [1, 32, 32, 64],
            [2, 64, 128, 256],
            [4, 128, 64, 512],
            [8, 256, 256, 128],
            [16, 512, 512, 256],
            [32, 1024, 1024, 128]
        ]
        
        for shape in test_shapes:
            batch_size, p1, p2, m = shape
            print(f"\n形状: batch_size={batch_size}, p1={p1}, p2={p2}, m={m}")
            
            # 生成随机输入数据
            x1 = torch.randn(batch_size, p1, m).to(dtype)
            x2 = torch.randn(batch_size, p2, m).to(dtype)
            
            # 对于fp16，缩小数值范围以避免溢出
            if dtype == torch.float16:
                x1 = x1 * 0.1
                x2 = x2 * 0.1
            
            compute_modes = ['use_mm_for_euclid_dist_if_necessary']
            if p == 2.0:
                compute_modes.extend(['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist'])
            
            for mode in compute_modes:
                print(f"\ncompute_mode: {mode}")
                test_result = run_accuracy_test(x1, x2, p, mode, device_cpu, device_npu, rtol, atol)
                logger.log_accuracy_test(p, dtype, shape, mode, test_result)

def run_accuracy_test(x1, x2, p, compute_mode, device_cpu, device_npu, rtol, atol):
    """运行单个准确度测试并返回结果字典"""
    try:
        # CPU测试
        x1_cpu = x1.to(device_cpu)
        x2_cpu = x2.to(device_cpu)
        
        # 确保CPU上使用fp32进行计算
        if x1_cpu.dtype != torch.float32:
            x1_cpu = x1_cpu.to(torch.float32)
            x2_cpu = x2_cpu.to(torch.float32)
        
        start_time = time.time()
        result_cpu = torch.cdist(x1_cpu, x2_cpu, p=p, compute_mode=compute_mode)
        cpu_time = time.time() - start_time
        
        # NPU测试
        x1_npu = x1.to(device_npu)
        x2_npu = x2.to(device_npu)
        
        # NPU预热
        _ = custom_cdist(x1_npu, x2_npu, p=p, compute_mode=compute_mode)
        torch.npu.synchronize()
        
        start_time = time.time()
        result_npu = custom_cdist(x1_npu, x2_npu, p=p, compute_mode=compute_mode)
        torch.npu.synchronize()
        npu_time = time.time() - start_time
        
        # 将结果移回CPU进行比较
        result_npu_cpu = result_npu.to(device_cpu)
        if result_npu_cpu.dtype != torch.float32:
            result_npu_cpu = result_npu_cpu.to(torch.float32)
        
        # 计算误差
        max_abs_diff = torch.max(torch.abs(result_cpu - result_npu_cpu)).item()
        max_rel_diff = torch.max(torch.abs((result_cpu - result_npu_cpu) / 
                                         (torch.abs(result_cpu) + 1e-7))).item()
        
        # 检查是否在误差范围内
        is_close = torch.allclose(result_cpu, result_npu_cpu, rtol=rtol, atol=atol)
        
        return {
            'status': 'passed' if is_close else 'failed',
            'max_abs_error': float(max_abs_diff),
            'max_rel_error': float(max_rel_diff),
            'cpu_time': float(cpu_time),
            'npu_time': float(npu_time),
            'speedup': float(cpu_time/npu_time)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }

def test_cdist_performance(logger, device_cpu='cpu', device_npu='npu', p=2.0, num_runs=10):
    """
    测试自定义cdist算子的性能
    
    参数:
        device_cpu: CPU设备
        device_npu: NPU设备
        p: p范数的p值
        num_runs: 每个测试运行的次数
    """
    print(f"\n测试cdist算子性能 (p={p}, 运行{num_runs}次):")
    
    # 测试不同规模的输入
    test_shapes = [
        # [batch_size, p1, p2, m]
        [1, 32, 32, 64],
        [2, 64, 128, 256],
        [4, 128, 64, 512],
        [8, 256, 256, 128],
        [16, 512, 512, 256],  # 大规模测试
        [32, 1024, 1024, 128] # 超大规模测试
    ]
    
    # 测试不同compute_mode（仅在p=2时）
    compute_modes = ['use_mm_for_euclid_dist_if_necessary']
    if p == 2.0:
        compute_modes.extend(['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist'])
    
    for shape in test_shapes:
        batch_size, p1, p2, m = shape
        print(f"\n形状: batch_size={batch_size}, p1={p1}, p2={p2}, m={m}")
        
        # 生成随机输入数据
        x1 = torch.randn(batch_size, p1, m)
        x2 = torch.randn(batch_size, p2, m)
        
        for mode in compute_modes:
            print(f"\ncompute_mode: {mode}")
            
            # CPU测试
            x1_cpu = x1.to(device_cpu)
            x2_cpu = x2.to(device_cpu)
            
            # 预热
            _ = torch.cdist(x1_cpu, x2_cpu, p=p, compute_mode=mode)
            
            # 计时
            cpu_times = []
            for _ in range(num_runs):
                torch.cuda.synchronize() if device_cpu == 'cuda' else None
                start_time = time.time()
                _ = torch.cdist(x1_cpu, x2_cpu, p=p, compute_mode=mode)
                torch.cuda.synchronize() if device_cpu == 'cuda' else None
                cpu_times.append(time.time() - start_time)
            
            # NPU测试
            try:
                x1_npu = x1.to(device_npu)
                x2_npu = x2.to(device_npu)
                
                # 预热
                _ = custom_cdist(x1_npu, x2_npu, p=p, compute_mode=mode)
                
                # 计时
                npu_times = []
                for _ in range(num_runs):
                    torch.npu.synchronize()
                    start_time = time.time()
                    _ = custom_cdist(x1_npu, x2_npu, p=p, compute_mode=mode)
                    torch.npu.synchronize()
                    npu_times.append(time.time() - start_time)
                
                # 计算统计数据
                cpu_mean = np.mean(cpu_times)
                cpu_std = np.std(cpu_times)
                npu_mean = np.mean(npu_times)
                npu_std = np.std(npu_times)
                
                perf_result = {
                    'cpu_time': {
                        'mean': float(cpu_mean),
                        'std': float(cpu_std)
                    },
                    'npu_time': {
                        'mean': float(npu_mean),
                        'std': float(npu_std)
                    },
                    'speedup': float(cpu_mean/npu_mean)
                }
                logger.log_performance_test(p, shape, mode, perf_result)
                
            except Exception as e:
                print(f"NPU测试失败: {e}")

def test_different_p_values(logger, device_cpu='cpu', device_npu='npu'):
    """测试不同p值的cdist算子"""
    # 根据PyTorch文档测试不同的p值
    p_values = [
        0,              # 汉明距离
        1,              # 曼哈顿距离
        2,              # 欧氏距离
        2.5,            # 分数p范数
        float('inf'),   # 无穷范数
    ]
    
    for p in p_values:
        test_cdist_accuracy(logger, device_cpu, device_npu, p)

def test_scipy_equivalence(logger):
    """测试与scipy.spatial.distance.cdist的等价性"""
    try:
        from scipy.spatial.distance import cdist
        import numpy as np
        
        print("\n测试与scipy.spatial.distance.cdist的等价性:")
        
        # 生成测试数据，确保使用float32类型
        x1 = torch.randn(3, 4, dtype=torch.float32)
        x2 = torch.randn(5, 4, dtype=torch.float32)
        
        # 测试不同的p值
        for p in [1, 2, float('inf')]:
            print(f"\np = {p}")
            
            # PyTorch结果
            torch_result = torch.cdist(x1, x2, p=p).to(torch.float32)
            
            # 将数据转换为numpy时确保使用float32
            x1_np = x1.numpy().astype(np.float32)
            x2_np = x2.numpy().astype(np.float32)
            
            # Scipy结果
            if p == float('inf'):
                scipy_result = cdist(x1_np, x2_np, lambda u, v: np.abs(u - v).max()).astype(np.float32)
            elif p == 0:
                scipy_result = (cdist(x1_np, x2_np, 'hamming') * x1.shape[1]).astype(np.float32)
            else:
                scipy_result = cdist(x1_np, x2_np, 'minkowski', p=p).astype(np.float32)
            
            # 将scipy结果转换为torch tensor，确保使用float32
            scipy_result = torch.from_numpy(scipy_result)
            
            # 比较结果前确保数据类型一致
            torch_result = torch_result.to(torch.float32)
            scipy_result = scipy_result.to(torch.float32)
            
            # 比较结果
            try:
                is_close = torch.allclose(torch_result, scipy_result, rtol=1e-5, atol=1e-5)
                test_result = {
                    'is_equivalent': is_close,
                    'max_abs_error': float(torch.max(torch.abs(torch_result - scipy_result)).item()),
                    'max_rel_error': float(torch.max(torch.abs((torch_result - scipy_result) / 
                                                             (torch.abs(torch_result) + 1e-7))).item()),
                    'pytorch_result_type': torch_result.dtype,
                    'scipy_result_type': scipy_result.dtype,
                    'pytorch_result_shape': torch_result.shape,
                    'scipy_result_shape': scipy_result.shape
                }
                logger.log_scipy_test(p, test_result)
            except Exception as e:
                print(f"比较结果时出错: {str(e)}")
                print(f"PyTorch结果类型: {torch_result.dtype}")
                print(f"Scipy结果类型: {scipy_result.dtype}")
                print(f"PyTorch结果形状: {torch_result.shape}")
                print(f"Scipy结果形状: {scipy_result.shape}")
    
    except ImportError:
        print("未安装scipy，跳过scipy等价性测试")
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置设备
    device_cpu = 'cpu'
    device_npu = 'npu:0'  # 假设使用第一个NPU设备
    
    print("=" * 50)
    print("cdist算子测试")
    print("=" * 50)
    
    # 创建结果记录器
    logger = TestResultLogger()
    
    # 测试准确度
    print("\n测试不同p值的准确度:")
    test_different_p_values(logger, device_cpu, device_npu)
    
    # 测试性能
    print("\n测试欧氏距离(p=2)的性能:")
    test_cdist_performance(logger, device_cpu, device_npu, p=2.0, num_runs=10)
    
    # 测试其他常用p值的性能
    print("\n测试曼哈顿距离(p=1)的性能:")
    test_cdist_performance(logger, device_cpu, device_npu, p=1.0, num_runs=5)
    
    # 测试与scipy的等价性
    test_scipy_equivalence(logger)
    
    # 保存结果
    logger.save_results()
