import torch
# We are going to create and see some random tensors

# They're important because the way many neural networks learn is that they start with
# tensors full of random numbers

# Create a random tensor of size (3,4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)

# Create a random tensor witch similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3, 224, 224)) # height, width, colour channels (R, G, B)
print(random_image_size_tensor)
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)