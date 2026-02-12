"""
GPU Verification Utility
Checks CUDA availability and GPU compatibility for fine-tuning
"""

import torch
import sys


def check_gpu():
    """
    Verify GPU availability and specifications
    
    Returns:
        bool: True if GPU is available and compatible
    """
    print("=" * 60)
    print("GPU Compatibility Check")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        print("   Please install CUDA-enabled PyTorch")
        return False
    
    print("✅ CUDA is available")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    print(f"   GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        print(f"\n   GPU {i}: {gpu_name}")
        print(f"   Total Memory: {gpu_memory:.2f} GB")
        
        # Check if memory is sufficient (minimum 4GB for QLoRA)
        if gpu_memory < 3.5:
            print(f"   ⚠️  Warning: GPU memory may be insufficient for training")
            print(f"      Recommended: ≥4GB, Available: {gpu_memory:.2f}GB")
        else:
            print(f"   ✅ GPU memory is sufficient for QLoRA training")
    
    # Test GPU allocation
    try:
        print("\n   Testing GPU allocation...")
        test_tensor = torch.randn(1000, 1000).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        print("   ✅ GPU allocation test passed")
    except Exception as e:
        print(f"   ❌ GPU allocation test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("GPU check completed successfully!")
    print("=" * 60)
    
    return True


def get_gpu_info():
    """
    Get detailed GPU information as dictionary
    
    Returns:
        dict: GPU information
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpus": []
        }
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count
        })
    
    return {
        "available": True,
        "cuda_version": torch.version.cuda,
        "gpu_count": len(gpus),
        "gpus": gpus
    }


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
