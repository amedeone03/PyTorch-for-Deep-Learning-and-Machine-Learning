import torch

# Indexing (selecting data from tensors)
x = torch.arange(1, 10).reshape(1, 3, 3)
print('tensor:', x, 'shape:', x.shape)

# Let's index on our new tensor
print(x[0])
print(x[0][0])
print(x[0][0][0])

# You can also use ":" to select "all" of a target dimension
print(x[:, 0])
print(x[:, :, 1])

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0, 0, :])

# Exercise: Index on x to return 9 and index on x to return 3, 6, 9
print('Solve the exercise', x[0][2][2], x[:, :, 2])