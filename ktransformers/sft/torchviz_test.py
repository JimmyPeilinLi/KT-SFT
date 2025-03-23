import torch
import torch.nn as nn
from torchviz import make_dot

# 定义一个简单的神经网络架构
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()

# 生成一个随机输入
input_tensor = torch.randn(1, 10)

# 前向传播
output = model(input_tensor)

# 使用 torchviz 进行可视化
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render('simple_net', format='svg', cleanup=True)    