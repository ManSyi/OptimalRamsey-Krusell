import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from OptimalConstraintKrusell import (Calibration, Simulation, WorkSpace, NeuralNetwork, LossFunction,
                                      xi_fun, xi_fun_with_grad)

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# 检查CUDA是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
print(f"使用设备: {device}")

torch.set_default_device('cpu')

# 设置随机种子以确保可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

cal = Calibration()
sml = Simulation()
# 初始化模型并将其移动到GPU
model = NeuralNetwork(cal).to(device)
ws  = WorkSpace(model, cal.Nj, cal.Nk)



# 1. 创建示例数据集 (在实际应用中，您会加载自己的数据)




x_data = sml.sample_index


# 划分训练集和测试集
train_size = int(0.8 * len(x_data))
test_size = len(x_data) - train_size
x_train, x_test = torch.split(x_data, [train_size, test_size])

# 创建DataLoader以便批量处理
batch_size = cal.batch_size
train_dataset = TensorDataset(x_train)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator(device='cpu'))

test_dataset = TensorDataset(x_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cpu'))

# 3. 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=cal.learning_rate)  # Adam优化器

# 4. 训练模型
num_epochs = cal.num_epochs
train_losses = []
test_losses = []

criterion = nn.MSELoss()
loss_fun = LossFunction()

sml.initial_sample()
ws.set_exo_shocks(sml)
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    for batch_x in train_loader:
        # 前向传播
        loss = loss_fun(xi_fun, criterion, xi_fun_with_grad, torch.stack(batch_x, 0).squeeze(), cal, ws, sml)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 测试阶段
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch_x in test_loader:
            loss = loss_fun(xi_fun, criterion, xi_fun_with_grad, torch.stack(batch_x, 0).squeeze(), cal, ws, sml)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # 每100个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# 绘制结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()



plt.tight_layout()
plt.show()