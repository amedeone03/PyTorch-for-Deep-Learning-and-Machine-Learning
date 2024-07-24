import torch
print(torch.__version__)

# Create a scalar 
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)

# Create a vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)

# Create a Matrix
Matrix = torch.tensor([[7, 8],
                      [9, 10]])
print(Matrix)
print(Matrix.ndim)

# Create a tensor
TENSOR = torch.tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
print(TENSOR)
print(TENSOR.ndim)

# What about the shape of a Tensor?
print(TENSOR.shape)