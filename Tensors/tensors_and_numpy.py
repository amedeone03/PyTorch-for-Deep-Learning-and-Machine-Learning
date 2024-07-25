import torch
import numpy as np

# Convert numpy array to PyTorch tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)

# Convert PyTorch tensor to numpy array
tensor1 = torch.arange(1.0, 9.0)
array1 = tensor1.numpy()
print(tensor1, array1)
