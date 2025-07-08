import torch
import matplotlib.pyplot as plt

# 获取PyTorch版本
pytorch_version = torch.__version__
print(f"{pytorch_version}")

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"{cuda_available}")

# 如果CUDA可用，显示GPU内存和名称
if cuda_available:
    # 获取内存信息（以MB为单位）
    memory_bytes = torch.cuda.get_device_properties(0).total_memory
    memory_mb = memory_bytes // (1024 * 1024)
    print(f"{memory_mb}")

    # 获取GPU名称
    device_name = torch.cuda.get_device_name(0)
    print(f"{device_name}")
else:
    print("CUDA not available")
    print("No GPU detected")