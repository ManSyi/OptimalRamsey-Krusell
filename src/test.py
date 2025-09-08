import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置随机种子以确保可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# 1. 创建示例数据集 (在实际应用中，您会加载自己的数据)
def create_sample_data(n_samples=10000):
    x = torch.randn(n_samples, 1)  # 输入特征
    y = 2.5 * x ** 3 + x ** 2 + 1.2 * x -0.56 # 线性关系加上噪声
    return x, y


# 生成数据
x_data, y_data = create_sample_data(1000)

# 将数据移动到GPU（如果可用）
x_data = x_data.to(device)
y_data = y_data.to(device)

# 划分训练集和测试集
train_size = int(0.8 * len(x_data))
test_size = len(x_data) - train_size
x_train, x_test = torch.split(x_data, [train_size, test_size])
y_train, y_test = torch.split(y_data, [train_size, test_size])

# 创建DataLoader以便批量处理
batch_size = 64
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# 初始化模型并将其移动到GPU
model = NeuralNetwork().to(device)

print("当前使用设备：", torch.cuda.current_device())
# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 4. 训练模型
num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

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
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# 5. 评估模型
model.eval()
with torch.no_grad():
    # 在测试集上进行预测
    test_predictions = model(x_test).cpu().numpy()
    x_test_cpu = x_test.cpu().numpy()
    y_test_cpu = y_test.cpu().numpy()

# 绘制结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_test_cpu, y_test_cpu, label='True Data', alpha=0.5)
plt.scatter(x_test_cpu, test_predictions, label='Predictions', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Predictions vs True Data')
plt.legend()

plt.tight_layout()
plt.show()