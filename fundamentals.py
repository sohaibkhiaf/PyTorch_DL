import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



print(f"Versions ==========================================")
print(f"torch version: {torch.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {plt.matplotlib.__version__}")
print("\n\n")




print(f"Scalar ==========================================")
scalar= torch.tensor(7)
print(f"scalar: {scalar}")
print(f"scalar ndim: {scalar.ndim}")
print(f"scalar item: {scalar.item()}")
print("\n\n")



print(f"Vector ==========================================")
vector = torch.tensor([7, 8])
print(f"vector: {vector}")
print(f"vector ndim: {vector.ndim}")
print(f"vector shape: {vector.shape}")
print("\n\n")




print(f"Matrix ==========================================")
MATRIX= torch.tensor([[7, 8], [9, 10]])
print(f"MATRIX: {MATRIX}")
print(f"MATRIX ndim: {MATRIX.ndim}")
print(f"MATRIX shape: {MATRIX.shape}")
print(f"MATRIX[0]: {MATRIX[0]}")
print(f"MATRIX[1]: {MATRIX[1]}")
print("\n\n")




print(f"Tensor ==========================================")
TENSOR= torch.tensor([[[1, 2, 3],
                       [3, 6, 9],
                       [2, 4, 6], ]])
print(f"TENSOR: {TENSOR}")
print(f"TENSOR ndim: {TENSOR.ndim}")
print(f"TENSOR shape: {TENSOR.shape}")
print(f"TENSOR[0]: {TENSOR[0]}")
print("\n\n")




print(f"Random tensor ==========================================")
random_tensor= torch.rand(3, 4)
print(f"random_tensor shape: {random_tensor.shape}")
print(f"random_tensor ndim: {random_tensor.ndim}")
print("\n\n")




print(f"Random image ==========================================")
random_image = torch.rand(size=(224, 224, 3))
print(f"random_image shape: {random_image.shape}")
print(f"random_image ndim: {random_image.ndim}")
print("\n\n")




print(f"Zeros / ones ==========================================")
zeros = torch.zeros(size=(3, 4))
ones = torch.ones(size=(3, 4))

print(f"zeros: {zeros}")
print(f"zeros shape: {zeros.shape}")
print(f"zeros ndim: {zeros.ndim}")
print(f"ones: {ones}")
print(f"ones shape: {ones.shape}")
print(f"ones ndim: {ones.ndim}")
print("\n\n")




print(f"Arange ==========================================")
one_to_ten = torch.arange(start=1, end=11, step=1)

print(f"one_to_ten= {one_to_ten}")
print("\n\n")




print(f"Like ==========================================")
ten_zeros = torch.zeros_like(one_to_ten)
ten_ones = torch.ones_like(one_to_ten)

print(f"ten_zeros= {ten_zeros}")
print(f"ten_ones= {ten_ones}")
print("\n\n")



print(f"Tensor datatypes ==========================================")
float_32_tensor = torch.tensor([3.0, 6.0 , 9.0],
                               dtype=None,
                               device=None ,
                               requires_grad=False)

int_32_tensor = torch.tensor([3, 6,9 ], dtype=torch.long)

float_16_tensor = float_32_tensor.type(torch.float16)

print(f"float_32_tensor dtype: {float_32_tensor.dtype}")
print(f"int_32_tensor dtype: {int_32_tensor.dtype}")
print(f"float_16_tensor dtype: {float_16_tensor.dtype}")
print("\n\n")




print(f"Tensor device ==========================================")
# tensor device
some_tensor = torch.rand(5, 8)
print(f"some_tensor is on: {some_tensor.device} " )
print("\n\n")




print(f"Tensor addition ==========================================")
tensor = torch.tensor([1,2 ,3])
tensor_after_addition = torch.add(tensor, 3) # same as tensor+ 3
print(f"tensor: {tensor}")
print(f"tensor_after_addition: {tensor_after_addition}")
print("\n\n")




print(f"Substraction ==========================================")
tensor_after_substraction = torch.sub(tensor, 10) # same as tensor- 10
print(f"tensor_after_substraction: {tensor_after_substraction}")
print("\n\n")




print(f"Multiplication ==========================================")
tensor_after_multiplication = torch.mul(tensor, 5) # tensor* 5
print(f"tensor_after_multiplication: {tensor_after_multiplication}")
print("\n\n")




print(f"Mathematical operations ==========================================")
# another way to perform mathematical operations on tensors
tensor_add = tensor+ 3
tensor_sub = tensor - 10
tensor_mul = tensor * 2
print(f"tensor_add: {tensor_add}")
print(f"tensor_sub: {tensor_sub}")
print(f"tensor_mul: {tensor_mul}")
print("\n\n")




print(f"Matrix multiplication ==========================================")
tensor_a = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_b = torch.tensor([[7, 8], [9, 10], [11, 12]])

print(f"tensor_a: {tensor_a}")
print(f"tensor_b: {tensor_b}")
print(f"tensor_a shape: {tensor_a.shape}")
print(f"tensor_b shape: {tensor_b.shape}")
print(f"tensor_b.T: {tensor_b.T}")
print(f"tensor_b.T.shape: {tensor_b.T.shape}")

tensor_mul = torch.matmul(tensor_a, tensor_b.T)
print(f"tensor_mul (torch.matmul(tensor_a, tensor_b.T): {tensor_mul}")
print(f"tensor_mul shape: {tensor_mul.shape}")
print("\n\n")




print(f"Tensor aggregation ==========================================")
x= torch.arange(0, 100, 10)

print(f"x: {x}")
print(f"x shape: {x.shape}")
print(f"x min: {torch.min(x)}") # same as x.min()
print(f"x max: {torch.max(x)}") # same as x.max()
print(f"x mean: {torch.mean(x.type(torch.float32))}") # same as x.type(torch.float32).mean()
print(f"x sum: {torch.sum(x)}") # same as x.sum()
print(f"x argmin: {torch.argmin(x)}") # same as x.argmin()
print(f"x argmax: {torch.argmax(x)}") # same as x.argmax()
print("\n\n")





print(f"Reshaping ==========================================")
x= torch.arange(1., 13.)
x_reshaped = x.reshape(3, 4)
print(f"x: {x}")
print(f"x shape: {x.shape}")
print(f"x reshaped: {x_reshaped}")
print(f"x reshaped shape: {x_reshaped.shape}")
print("\n\n")




print(f"View ==========================================")
v= x_reshaped.view(3, 4)
print(f"v: {v}")
print(f"v shape: {v.shape}")
print("\n")
print("view of a tensor shares the same memory as the original tensor: ")
v[0, :] = -1
print(f"v: {v}")
print(f"x_reshaped: {x_reshaped}")
print("\n\n")




print(f"Stack ==========================================")
x_stacked = torch.stack([x, x, x], dim=0) # dim=0 = vstack, dim=1 = hstack
print(f"x_stacked: {x_stacked}")
print(f"x_stacked shape: {x_stacked.shape}")
print("\n\n")




print(f"Squeezing ==========================================")
x= torch.rand(2, 1, 3, 4,1 )
x_squeezed = x.squeeze()
print(f"x shape: {x.shape}")
print(f"x_squeezed shape: {x_squeezed.shape}")
print("\n\n")





print(f"Unsqueezing ==========================================")
x= x_squeezed
x_unsqueezed = x.unsqueeze(dim=0)
print(f"x shape: {x_squeezed.shape}")
print(f"x_unsqueezed shape: {x_unsqueezed.shape}")
print("\n\n")




print(f"Permuting ==========================================")
x = x_unsqueezed
x_permuted = x.permute(3, 2, 1, 0)

print(f"x shape: {x_unsqueezed.shape}")
print(f"x_permuted shape: {x_permuted.shape}")
print("\n\n")




print(f"Indexing ==========================================")
x= torch.arange(1, 10).reshape(1, 3, 3)

print(f"x: {x}")
print(f"x shape: {x.shape}")
print(f"x[0]: {x[0]}")
print(f"x[0].shape: {x[0].shape}")
print(f"x[0, 0]: {x[0, 0]}")
print(f"x[0, 0].shape: {x[0, 0].shape}")
print(f"x[0, 0, 0]: {x[0, 0, 0]}")
print(f"x[0, 0, 0].shape: {x[0, 0, 0].shape}")
print(f"x[:,:,2]: {x[:,:,2]}")
print(f"x[:,:,2].shape: {x[:,:,2].shape}")
print("\n\n")




print(f"Convert numpy array to torch.Tensor ==========================================")
np_arr1 = np.arange(1., 8.)
torch_tensor1 = torch.from_numpy(np_arr1)
print(f"np_arr1: {np_arr1}")
print(f"torch_tensor1: {torch_tensor1}")
print("\n\n")




print(f"Convert torch.Tensor to numpy array ==========================================")
torch_tensor2 = torch.arange(1., 8.)
np_arr2 = torch_tensor2.numpy()
print(f"torch_tensor2: {torch_tensor2}")
print(f"np_arr2: {np_arr2}")
print("\n\n")




print(f"Manual seed ==========================================")
torch.manual_seed(42)
tensor_a = torch.rand(3, 4)
torch.manual_seed(42)
tensor_b = torch.rand(3, 4)
print(f"Test manual seed: {tensor_a == tensor_b}")
print("\n\n")



print(f"Device agnostic code ==========================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
num_devices = torch.cuda.device_count()
print(f"number of devices: {num_devices}")
print("\n\n")



print(f"Create tensor on cpu ==========================================")
tensor = torch.tensor([1, 2,3 ], device="cpu")
print(f"tensor is on: {tensor.device}")
print("\n\n")



print(f"Move tensor to gpu if available ==========================================")
tensor_on_gpu = tensor.to(device)
print(f"tensor_on_gpu is on: {tensor_on_gpu.device}")
print("\n\n")




print(f"Move tensor back to cpu + convert to numpy ==========================================")
#numpy, matplotlib and pandas work strictly on cpu
tensor_back_to_cpu = tensor_on_gpu.cpu().numpy()
print(f"tensor_back_to_cpu: {tensor_back_to_cpu}")
print(f"tensor_back_to_cpu dtype: {tensor_back_to_cpu.dtype}")
print("\n\n")

