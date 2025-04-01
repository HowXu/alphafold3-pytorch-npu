import numpy as np
import torch
import torch.nn as nn
import torch_npu
from parameter import Parameter as P # 导入自定义Parameter类

# 设置随机种子保证可重复性
torch.manual_seed(42)
if torch.npu.is_available():
    torch_npu.npu.manual_seed(42)

def compare_results(cpu_tensor, npu_tensor, name="Tensor", atol=1e-5):
    """增强版结果对比函数"""
    npu_tensor_cpu = npu_tensor.cpu().detach()
    cpu_tensor_detached = cpu_tensor.detach()
    
    max_diff = torch.max(torch.abs(npu_tensor_cpu - cpu_tensor_detached)).item()
    mean_diff = torch.mean(torch.abs(npu_tensor_cpu - cpu_tensor_detached)).item()
    
    print(f"\n[{name}]")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if torch.allclose(npu_tensor_cpu, cpu_tensor_detached, atol=atol):
        print(f"✅ Results match (atol={atol})")
    else:
        print(f"❌ Results exceed tolerance (atol={atol})")
    return max_diff

def test_parameter_npu_cpu_consistency():
    # 确保使用正确的设备
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    print(f"Testing device: {device.type.upper()}")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 使用自定义Parameter并显式指定设备
            self.param = P(torch.randn(128, 256, device=device))  # 关键修改点1
            self.fc = nn.Linear(256, 128)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # 添加设备检查断言
            assert x.device == self.param.device, \
                f"Input device {x.device} vs param device {self.param.device}"
            x = torch.matmul(x, self.param)
            x = self.fc(x)
            x = self.relu(x)
            return x

    # 初始化模型（显式设备分配）
    cpu_model = TestModel().to("cpu")
    npu_model = TestModel().to(device)
    
    # 精确参数同步（处理自定义Parameter）
    with torch.no_grad():
        for (cpu_name, cpu_param), (npu_name, npu_param) in zip(
            cpu_model.named_parameters(),
            npu_model.named_parameters()
        ):
            # 确保参数类型正确
            if isinstance(npu_param, P):
                npu_param.data = cpu_param.data.clone().to(device)  # 关键修改点2
            else:
                npu_param.copy_(cpu_param.data)
    
    # 创建测试数据（显式设备分配）
    x_cpu = torch.randn(32, 256, device="cpu", requires_grad=True)
    x_npu = x_cpu.detach().clone().to(device).requires_grad_(True)
    
    # 前向计算前验证设备一致性
    print("\n[Device Check]")
    print(f"Input device: {x_npu.device}")
    for name, param in npu_model.named_parameters():
        print(f"Parameter '{name}' device: {param.device}")
    
    # 执行前向传播
    cpu_output = cpu_model(x_cpu)
    npu_output = npu_model(x_npu)
    
    # 结果比较
    max_diff = compare_results(cpu_output, npu_output, "Forward Output", atol=1e-5)
    
    # 反向传播验证
    target = torch.randn_like(cpu_output)
    cpu_loss = torch.mean((cpu_output - target)**2)
    npu_loss = torch.mean(npu_output - target.to(device))**2
    
    cpu_loss.backward()
    npu_loss.backward()
    
    # 梯度比较
    compare_results(x_cpu.grad, x_npu.grad, "Input Gradients")
    for i, (cpu_param, npu_param) in enumerate(zip(cpu_model.parameters(), npu_model.parameters())):
        compare_results(cpu_param.grad, npu_param.grad, f"Param Grad {i}")
    
    return max_diff < 1e-5

if __name__ == "__main__":
    if not torch.npu.is_available():
        print("Warning: NPU device not available, testing CPU vs CPU")
    
    try:
        test_passed = test_parameter_npu_cpu_consistency()
    except RuntimeError as e:
        print(f"\n❌ Critical Error: {str(e)}")
        print("建议检查：")
        print("1. NPU驱动和PyTorch版本兼容性")
        print("2. 自定义Parameter的设备迁移逻辑")
        print("3. 张量设备一致性（使用tensor.device属性验证）")
        test_passed = False
    
    if test_passed:
        print("\n🎉 Test Passed: NPU and CPU results match within tolerance!")
    else:
        print("\n  Test Warning: Numerical differences detected")
