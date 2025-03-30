import re
import torch_npu
#from torch import cdist
import torch
from torch import nn

NPU = 1

def tensor_to_npu(obj):
    # 获取对象的类型名称
    #type_name = type(obj).__name__
    
    # 使用正则表达式匹配 Tensor 或 Tensor@xxxx 的形式
    #if re.match(r'^Tensor(@\w+)?$', type_name):
        # 这个写法肯定不是最好的，凑合用

    if NPU == 0:
        return obj

    if obj is None:
        raise ValueError("tensor_to_npu received a None object")
    if obj.device.type != "npu":
        return obj.npu()
    return obj

def tensor_to_npu_re(obj):
    # 获取对象的类型名称
    type_name = type(obj).__name__
    
    # 使用正则表达式匹配 Tensor 或 Tensor@xxxx 的形式
    if re.match(r'^Tensor(@\w+)?$', type_name):
        # 这个写法肯定不是最好的，凑合用
        if obj.device.type != "npu":
            return obj.npu()
    return obj  

def cdist_npu(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """
    自定义实现的cdist算子 使用基础算子组合实现
    针对NPU硬件进行优化
    
    参数:
        x1 (Tensor): 形状为 B P M 的输入张量
        x2 (Tensor): 形状为 B R M 的输入张量
        p (float): p范数的p值 默认为2.0 欧氏距离
        compute_mode (str): 计算模式 与原始cdist保持一致
    
    返回:
        Tensor: 形状为 B P R 的距离矩阵
    """
    # Cpu 情况断言
    if NPU != 0:
        # 这是值传递还是引用传递?
        return torch.cdist(x1,x2,p)
    
    # 原始NPU实现，保留作为参考
    # batch_size = x1.shape[0]
    # p1, m = x1.shape[1], x1.shape[2]
    # p2 = x2.shape[1]
    # 
    # # 检查输入数据类型
    # input_dtype = x1.dtype
    # compute_dtype = input_dtype
    # if input_dtype == torch.float16 and p == 2.0:
    #     # 对于fp16的欧氏距离计算，使用fp32提高精度
    #     compute_dtype = torch.float32
    #     x1 = x1.to(compute_dtype)
    #     x2 = x2.to(compute_dtype)
    # 
    # if p == 2.0:
    #     # 直接计算差值的平方和，避免矩阵乘法reshape问题
    #     x1_expanded = x1.unsqueeze(2)  # B×P×1×M
    #     x2_expanded = x2.unsqueeze(1)  # B×1×R×M
    #     result_squared = torch.sum((x1_expanded - x2_expanded) ** 2, dim=3)
    #     
    #     # 确保数值稳定性
    #     result_squared = torch.clamp(result_squared, min=0.0)
    #     result = torch.sqrt(result_squared)
    # else:
    #     # 对于其他p值的实现
    #     if p == 0:
    #         # 汉明距离优化
    #         x1_expanded = x1.unsqueeze(2)  # B×P×1×M
    #         x2_expanded = x2.unsqueeze(1)  # B×1×R×M
    #         result = torch.sum(torch.ne(x1_expanded, x2_expanded).to(compute_dtype), dim=3)
    #         
    #     elif p == 1:
    #         # 曼哈顿距离优化
    #         x1_expanded = x1.unsqueeze(2)  # B×P×1×M
    #         x2_expanded = x2.unsqueeze(1)  # B×1×R×M
    #         result = torch.sum(torch.abs(x1_expanded - x2_expanded), dim=3)
    #         
    #     elif p == float('inf'):
    #         # 无穷范数优化
    #         x1_expanded = x1.unsqueeze(2)  # B×P×1×M
    #         x2_expanded = x2.unsqueeze(1)  # B×1×R×M
    #         result = torch.max(torch.abs(x1_expanded - x2_expanded), dim=3)[0]
    #         
    #     else:
    #         # 一般p范数优化
    #         x1_expanded = x1.unsqueeze(2)  # B×P×1×M
    #         x2_expanded = x2.unsqueeze(1)  # B×1×R×M
    #         diff = torch.abs(x1_expanded - x2_expanded)
    #         if p < 1:
    #             # 对于p < 1的情况，先计算幂再求和，避免数值不稳定
    #             result = torch.sum(diff ** p, dim=3) ** (1/p)
    #         else:
    #             result = torch.sum(diff ** p, dim=3) ** (1/p)
    # 
    # # 恢复原始数据类型
    # if result.dtype != input_dtype:
    #     result = result.to(input_dtype)
    # 
    # return result
    
    # 针对NPU优化的新实现
    batch_size = x1.shape[0]
    p1, m = x1.shape[1], x1.shape[2]
    p2 = x2.shape[1]
    
    # 检查输入数据类型
    input_dtype = x1.dtype
    compute_dtype = input_dtype
    
    # 根据华为NPU文档，NPU对fp16有良好支持
    # 但对于精度敏感的计算，仍使用fp32
    if input_dtype != torch.float16 and input_dtype != torch.float32:
        compute_dtype = torch.float32
        x1 = x1.to(compute_dtype)
        x2 = x2.to(compute_dtype)
    
    # 分块计算的阈值，避免内存溢出
    BATCH_THRESHOLD = 32
    
    # 对于大批量数据，进行分块计算
    if batch_size > BATCH_THRESHOLD:
        results = []
        chunk_size = BATCH_THRESHOLD
        for i in range(0, batch_size, chunk_size):
            end = min(i + chunk_size, batch_size)
            chunk_x1 = x1[i:end]
            chunk_x2 = x2[i:end]
            
            # 递归调用处理小批量数据
            chunk_result = cdist_npu(chunk_x1, chunk_x2, p, compute_mode)
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)
    
    # 针对不同p值的优化计算
    if p == 2.0:
        # 欧氏距离计算
        # 根据NPU特性，选择合适的计算方法
        if compute_mode in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist'] and (p1 > 25 or p2 > 25):
            # 矩阵乘法方法，适用于较大规模计算
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            # 使用批量处理避免reshape带来的问题
            
            # 计算 ||x||^2
            x1_norm = torch.sum(x1 * x1, dim=2, keepdim=True)  # B×P×1
            
            # 计算 ||y||^2
            x2_norm = torch.sum(x2 * x2, dim=2, keepdim=True)  # B×R×1
            
            # 计算内积 <x,y>，使用批量循环避免大矩阵
            dot_products = torch.zeros((batch_size, p1, p2), device=x1.device, dtype=compute_dtype)
            
            for b in range(batch_size):
                # 使用matmul计算当前批次的内积，NPU对matmul有优化
                dot_products[b] = torch.matmul(x1[b], x2[b].transpose(0, 1))
            
            # 计算距离平方 ||x||^2 + ||y||^2 - 2<x,y>
            result_squared = x1_norm + x2_norm.transpose(1, 2) - 2 * dot_products
        else:
            # 直接计算方法，适用于小规模计算
            # 使用广播机制直接计算距离
            x1_expanded = x1.unsqueeze(2)  # B×P×1×M
            x2_expanded = x2.unsqueeze(1)  # B×1×R×M
            result_squared = torch.sum((x1_expanded - x2_expanded) ** 2, dim=3)
        
        # 确保数值稳定性
        result_squared = torch.clamp(result_squared, min=0.0)
        result = torch.sqrt(result_squared)
    else:
        # 其他p范数的优化计算
        x1_expanded = x1.unsqueeze(2)  # B×P×1×M
        x2_expanded = x2.unsqueeze(1)  # B×1×R×M
        
        if p == 0:
            # 汉明距离
            # 根据NPU文档，torch.ne在NPU上有支持
            not_equal = torch.ne(x1_expanded, x2_expanded)
            result = torch.sum(not_equal.to(compute_dtype), dim=3)
        elif p == 1:
            # 曼哈顿距离
            # 根据NPU文档，torch.abs在NPU上有良好支持
            result = torch.sum(torch.abs(x1_expanded - x2_expanded), dim=3)
        elif p == float('inf'):
            # 无穷范数
            # 根据NPU文档，torch.max在NPU上有支持
            result = torch.max(torch.abs(x1_expanded - x2_expanded), dim=3)[0]
        else:
            # 一般p范数
            diff = torch.abs(x1_expanded - x2_expanded)
            if p < 1:
                # 对于p < 1的情况，计算更稳定的方式
                result = torch.sum(diff ** p, dim=3) ** (1/p)
            else:
                result = torch.sum(diff ** p, dim=3) ** (1/p)
    
    # 恢复原始数据类型
    if result.dtype != input_dtype:
        result = result.to(input_dtype)
    
    return result