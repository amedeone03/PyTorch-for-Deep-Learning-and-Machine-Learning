import torch

# Torch defines various tensor types with CPU and GPU variants
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,
                               device = None,
                               requires_grad=False)
print(float_32_tensor)