import torch

# In this code, we will see some methods such that:
# Reshaping, staking, squeezing and unsqueezing tensors

# Create a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('The tensor is:', tensor)

# Reshape tensor to (3, 2)
reshaped_tensor = tensor.reshape(3, 2)
print("Reshaped Tensor: ", reshaped_tensor)

# Staking tensors along a specified dimension
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

stacked_tensor = torch.stack((tensor1, tensor2), dim=0)
print("Stacked Tensor: ", stacked_tensor)

# Squeezing a tensor
squeezed_tensor = torch.squeeze(tensor)
print("Squeezed Tensor: ", squeezed_tensor)

# Unsqueezing a tensor
unsqueezed_tensor = torch.unsqueeze(squeezed_tensor, dim=0)
print("Unsqueezed Tensor: ", unsqueezed_tensor)

# Note: Reshaping, stacking, squeezing, and unsqueezing tensors are fundamental operations in deep learning and machine learning. They help in manipulating the dimensions of tensors and performing various operations on them.
