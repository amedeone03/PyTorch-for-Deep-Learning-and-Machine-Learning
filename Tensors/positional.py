import torch

# Find the positional min and max values
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print('The tensor is:', x)
print('Positional minimum:', torch.min(x).item())
print('Positional maximum:', torch.max(x).item())

# Find the indices of the positional min and max values
print(x.argmin())
print(x.argmax())