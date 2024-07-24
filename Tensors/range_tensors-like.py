import torch
7
# Use torch.arange() 
x = torch.arange(1, 11)
print(x)

y = torch.arange(1, 1000, 77)
print(y)

# Creating tensors like
input_tensor = torch.tensor([1, 2, 3])
ten_zeros = torch.zeros_like(input_tensor)
print(ten_zeros)