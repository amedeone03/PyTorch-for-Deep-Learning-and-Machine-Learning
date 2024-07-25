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

# Tensor operations with the same datatype
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(int_32_tensor * float_32_tensor)

# To sum up, let's get the information from tensors
some_tensor = torch.rand(3, 4)
print(f"The tensor is: {some_tensor}")
print(f"Datatype of tensor: {some_tensor.dtype} ")
print(f"Shape of the tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")