# check_current_cuda.py
import torch
import sys

print("=" * 60)
print("当前PyTorch CUDA状态检查")
print("=" * 60)

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"Torchvision版本: 0.23.0")

# 检查编译信息
if hasattr(torch, '__config__'):
    print(f"编译选项: {torch.__config__.show()}")

# 检查CUDA
cuda_available = torch.cuda.is_available()
print(f"\nCUDA是否可用: {cuda_available}")

if cuda_available:
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")

    # 检查CUDA能力
    capability = torch.cuda.get_device_capability(0)
    print(f"计算能力: {capability[0]}.{capability[1]}")
else:
    print("\n❌ CUDA不可用！")
    print("虽然安装了CUDA版本的PyTorch，但无法访问GPU")
    print("\n可能原因：")
    print("1. PyTorch 2.8.0的CUDA 13.0是社区版本，不完整")
    print("2. 需要安装官方支持的CUDA版本（12.1或12.4）")

# 测试GPU计算
if cuda_available:
    print("\n测试GPU计算...")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x + y
        print("✅ GPU计算测试通过")
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")

print("\n" + "=" * 60)
print("建议：")
if not cuda_available:
    print("重新安装官方支持的CUDA 12.1版本的PyTorch")
    print(
        "命令：pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121")
print("=" * 60)