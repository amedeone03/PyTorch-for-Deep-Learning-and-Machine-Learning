import torch 

# Finding the min, max, mean and sum of a tensor

# Create a tensor
tensor = torch.arange(0, 100, 10)
print(tensor)
print(tensor.dtype)

# Find the minimum value
min_val = tensor.min()
print(min_val)

# Find the maximum value
max_val = tensor.max()
print(max_val)

# Find the mean value
# Be careful to convert the type of the tensor
mean_val = torch.mean(tensor.type(torch.float32))
print(mean_val)

# Find the sum of all values
sum_val = tensor.sum()
print(sum_val)