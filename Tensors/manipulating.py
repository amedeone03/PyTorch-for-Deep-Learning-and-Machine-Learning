import torch

# Manipulaing Tensors means tensor operations, which are:
# Addition
# Subtraction
# Multiplication
# Division
# Matrix multiplication

tensor = torch.tensor([1, 2, 3])
print(f"The tensor is: {tensor}")

# Addition
print(f"The addition is: {tensor + 100}")

# Subtraction
print(f"The subtraction is: {tensor - 100}")

# Multiplication
print(f"The multiplication is: {tensor * 10}")

# Try out PyTorch in-built functions
print(f"Try out PyTorch functions: {torch.mul(tensor, 10)}")

# Division

# Note: Division by zero is not allowed in PyTorch
# Try to divide the tensor by zero
try:
    print(f"The division is: {tensor / 0}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Matrix multiplication

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

print(f"The tensor1 is: {tensor1}")
print(f"The tensor2 is: {tensor2}")

# Matrix multiplication
result_matrix = torch.matmul(tensor1, tensor2)
print(f"The result of matrix multiplication is: {result_matrix}")

# Note: In PyTorch, matrix multiplication is performed by using the matmul() function

# Another example of matrix multiplication
tensor3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor4 = torch.tensor([[7,8], [9, 10], [11, 12]])
print(f"The tensor3 is: {tensor3}")
print(f"The tensor4 is: {tensor4}")

result_matrix_mul = torch.matmul(tensor3, tensor4)
print(f"The result of dot product is: {result_matrix_mul}")

# There are tow main rules that performing matrix multiplication must satisfy:
# 1. **Inner dimensions** must match
# 2. The resulting matrix has the shape of the **outer dimensions**