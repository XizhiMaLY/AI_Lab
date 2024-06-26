import numpy as np
import pandas as pd


data_path = '/home/agent_mxz/AI_Lab/sharedbike/hour.csv'  # 读取数据到内存，rides为一个dataframe对象
rides = pd.read_csv(data_path)

counts = rides['cnt'][:50]  # 截取数据
x = np.arange(len(counts))  # 获取变量x
y = np.array(counts) # 单车数量为y


import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden = self.linear1(x)
        hidden = self.sigmoid(hidden)
        output = self.linear2(hidden)
        return output



x = torch.tensor(np.arange(len(counts)), dtype=torch.float)
y = torch.tensor(np.array(counts), dtype=torch.float)

x = x.view(-1, 1)  # Reshape x to be a column vector
y = y.view(-1, 1)  # Reshape y to be a column vector

model = NeuralNet(input_size=1, hidden_size=10)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

losses = []

for i in range(100000):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % 10000 == 0:
        print('loss:', loss.item())

# Print the final learned weights and biases
print("Learned weights:", model.linear1.weight)
print("Learned biases:", model.linear1.bias)
print("Learned weights2:", model.linear2.weight)
print("Learned biases2:", model.linear2.bias)