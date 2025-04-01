import torch
import torch_npu

def custom_cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """
    自定义实现的cdist算子，使用基础算子组合实现
    
    参数:
        x1 (Tensor): 形状为 B×P×M 的输入张量
        x2 (Tensor): 形状为 B×R×M 的输入张量
        p (float): p范数的p值，默认为2.0（欧氏距离）
        compute_mode (str): 计算模式，与原始cdist保持一致
    
    返回:
        Tensor: 形状为 B×P×R 的距离矩阵
    """
    batch_size = x1.shape[0]
    p1, m = x1.shape[1], x1.shape[2]
    p2 = x2.shape[1]
    
    # 检查输入数据类型
    input_dtype = x1.dtype
    compute_dtype = input_dtype
    if input_dtype == torch.float16 and p == 2.0:
        # 对于fp16的欧氏距离计算，使用fp32提高精度
        compute_dtype = torch.float32
        x1 = x1.to(compute_dtype)
        x2 = x2.to(compute_dtype)
    
    if p == 2.0:
        # 欧氏距离的特殊处理
        if compute_mode in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist']:
            # 使用矩阵乘法优化大规模计算
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            x1_squared = torch.sum(x1 * x1, dim=2, keepdim=True)  # B×P×1
            x2_squared = torch.sum(x2 * x2, dim=2, keepdim=True)  # B×R×1
            
            # 重塑张量以使用批量矩阵乘法
            x1_reshaped = x1.reshape(batch_size * p1, m)
            x2_reshaped = x2.reshape(batch_size * p2, m)
            
            # 使用批量矩阵乘法计算内积
            dot_product = torch.matmul(x1_reshaped, x2_reshaped.t())
            dot_product = dot_product.reshape(batch_size, p1, p2)
            
            # 计算距离
            result_squared = x1_squared + x2_squared.transpose(1, 2) - 2 * dot_product
            
        else:
            # 直接计算差值的平方和
            x1_expanded = x1.unsqueeze(2)  # B×P×1×M
            x2_expanded = x2.unsqueeze(1)  # B×1×R×M
            result_squared = torch.sum((x1_expanded - x2_expanded) ** 2, dim=3)
        
        # 确保数值稳定性
        result_squared = torch.clamp(result_squared, min=0.0)
        result = torch.sqrt(result_squared)
        
    else:
        # 对于其他p值的实现
        if p == 0:
            # 汉明距离优化
            x1_expanded = x1.unsqueeze(2)  # B×P×1×M
            x2_expanded = x2.unsqueeze(1)  # B×1×R×M
            result = torch.sum(torch.ne(x1_expanded, x2_expanded).to(compute_dtype), dim=3)
            
        elif p == 1:
            # 曼哈顿距离优化
            x1_expanded = x1.unsqueeze(2)  # B×P×1×M
            x2_expanded = x2.unsqueeze(1)  # B×1×R×M
            result = torch.sum(torch.abs(x1_expanded - x2_expanded), dim=3)
            
        elif p == float('inf'):
            # 无穷范数优化
            x1_expanded = x1.unsqueeze(2)  # B×P×1×M
            x2_expanded = x2.unsqueeze(1)  # B×1×R×M
            result = torch.max(torch.abs(x1_expanded - x2_expanded), dim=3)[0]
            
        else:
            # 一般p范数优化
            x1_expanded = x1.unsqueeze(2)  # B×P×1×M
            x2_expanded = x2.unsqueeze(1)  # B×1×R×M
            diff = torch.abs(x1_expanded - x2_expanded)
            if p < 1:
                # 对于p < 1的情况，先计算幂再求和，避免数值不稳定
                result = torch.sum(diff ** p, dim=3) ** (1/p)
            else:
                result = torch.sum(diff ** p, dim=3) ** (1/p)
    
    # 恢复原始数据类型
    if result.dtype != input_dtype:
        result = result.to(input_dtype)
    
    return result
