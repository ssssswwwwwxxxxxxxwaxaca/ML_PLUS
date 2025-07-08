import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"显卡数量: {torch.cuda.device_count()}")
    print(f"当前显卡: {torch.cuda.get_device_name(0)}")
    print(f"显卡显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试CUDA张量
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("CUDA张量运算测试成功!")
else:
    print("CUDA不可用，使用CPU")