import torch

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像配置 
IMAGE_SIZE = 560  # Must be a multiple of 14
PATCH_SIZE = 14
