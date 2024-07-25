import torch

# See Precision on computer science on wikipedia
# Torch defines various tensor types with CPU and GPU variants
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,
                               device = None,
                               requires_grad=False)
print(float_32_tensor)

# Convert tensor to float 16
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

# Let's multiply the tensors
print(float_32_tensor * float_16_tensor)

# Tensor operations 
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(int_32_tensor * float_32_tensor)