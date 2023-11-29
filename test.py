# key_list = [key for key, _ in model.named_modules()]
import torch
# Tensor = torch.randn(3, 4)
# row_norms = torch.norm(Tensor, dim=1, keepdim=True)
# print(row_norms)

import torch

# 假设 tensor1 的形状为 (3, 5, 10)
# tensor1 = torch.randn(3, 5, 10)
#
# # 假设 tensor2 的形状为 (3, 1)
# tensor2 = torch.tensor([[1],
#                         [0],
#                         [2]])
#
# # 创建一个索引张量，以便在第二个维度上检索
# index = tensor2.unsqueeze(2).expand(-1, -1, 10)
#
# # 使用 torch.gather 进行检索
# output = torch.gather(tensor1, dim=1, index=index)

Tensor = torch.randn(3, 5, 10)
row_norms = torch.norm(Tensor, dim=2, keepdim=True)
norm_fracs = (Tensor ** 2) / row_norms ** 2
max_indices = norm_fracs.argmax(dim=1).unsqueeze(1)
selected_elements = torch.gather(Tensor, dim=1, index=max_indices)
print(Tensor)
print("----------------")
print(max_indices)
print("----------------")
print(selected_elements)
# print(tensor1)
# print(output)
# print(output.shape)

with open("/media/data/2/yx/model_toxic/my_project/results/vector_p_muti_gen/t(jjj).json", "w") as f:
    f.write("hello world")
