import torch

# Check for GPU access with PyTorch
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can access the GPU.")
else:
    print("CUDA is not available. PyTorch cannot access the GPU.")
