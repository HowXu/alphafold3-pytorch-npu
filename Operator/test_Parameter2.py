import torch
from torch.nn.parameter import Parameter, UninitializedParameter

import torch_npu

def check_migration():
    # 检查是否有可用的NPU设备
    if not torch.npu.is_available():
        print("NPU device is not available.")
        return False

    # 检查 Parameter 是否在 NPU 上
    param = Parameter().npu()
    if param.device.type != 'npu':
        print("Parameter is not on NPU.")
        return False
    else:
        print("Parameter is successfully on NPU.")

    # 检查 UninitializedParameter 是否在 NPU 上
    uninit_param = UninitializedParameter()
    uninit_param.materialize((10, 10),device=torch.device("npu")) # 需要先 materialize 才能检查设备
    #uninit_param.npu()  
    if uninit_param.device.type != 'npu':
        print(uninit_param.device.type)
        print("UninitializedParameter is not on NPU.")
        return False
    else:
        print("UninitializedParameter is successfully on NPU.")

    return True

if __name__ == "__main__":
    if check_migration():
        print("Migration to Ascend PyTorch is successful.")
    else:
        print("Migration to Ascend PyTorch failed.")