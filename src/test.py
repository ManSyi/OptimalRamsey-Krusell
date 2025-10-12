import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from OptimalConstraintKrusell import (Calibration, Simulation, WorkSpace,  initial_value_fun,
                                       IndexedDataset, LossFunction, InitializedModel, DeepSet,
                                      xi_fun, xi_fun_with_grad, adjust_learning_rate)

cal = Calibration()
torch.set_default_dtype(torch.float64)
device = cal.device
print(f"使用设备: {device}")

torch.set_default_device(cal.device_str)

# 设置随机种子以确保可重复性
torch.manual_seed(cal.seed_torch)
torch.cuda.manual_seed(cal.seed_torch)


sml = Simulation()
sml.initial_sample()
sml.set_exo_shocks()
sml.initial_value_fun()
# 初始化模型并将其移动到GPU

n_phi_layers = 2  # number of hidden layers in phi (set by main)
n_rho_layers = 4  # number of hidden layers in rho (set by main)
phi_hidden_dim = 64
phi_out_dim = 32
rho_hidden_dim = 128

res_net = DeepSet(g_dim=1,
                    phi_hidden_dim=phi_hidden_dim,
                    phi_out_dim=phi_out_dim,
                    rho_hidden_dim=rho_hidden_dim,
                    n_phi_layers=n_phi_layers,
                    n_rho_layers=n_rho_layers).to(device)

# model =  InitializedModel(res_net, initial_value_fun)
model = res_net
ws  = WorkSpace(model)


z = sml.z0
a = sml.a0
g = sml.g0
A = sml.A0


# 划分训练集和测试集
# train_size = int(0.8 * cal.Ns)
# test_size = cal.Ns - train_size
#
# z_train, z_test = torch.split(z, [train_size, test_size])
# a_train, a_test = torch.split(a, [train_size, test_size])
# g_train, g_test = torch.split(g, [train_size, test_size])
# A_train, A_test = torch.split(A, [train_size, test_size])

# 创建DataLoader以便批量处理
batch_size = cal.batch_size
# train_dataset = TensorDataset(z_train, a_train, g_train, A_train)
train_dataset = TensorDataset(z, a, g, A)

train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=batch_size, shuffle=True,
                          generator=torch.Generator(device=cal.device_str))

# test_dataset = TensorDataset(z_test, a_test, g_test, A_test)
#
#
# test_loader = DataLoader(IndexedDataset(test_dataset), batch_size=batch_size, shuffle=False, generator=torch.Generator(device=cal.device_str))

# 3. 定义损失函数和优化器
lr = cal.learning_rate
optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

# 4. 训练模型
num_epochs = cal.num_epochs
train_losses = []
# test_losses = []

criterion = nn.MSELoss()
loss_fun = LossFunction()
train_batch_num = len(train_loader)
np.savetxt("xi_fun_start", xi_fun(model, z, a,g,A).cpu().numpy(), delimiter=",")
epoch_update_learn = 5
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    current_sample = 0
    for batch, (z,a,g,A,index) in enumerate(train_loader):
        # 前向传播
        loss = loss_fun(xi_fun, criterion, xi_fun_with_grad, index, z, a, g,A, ws, sml)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        current_sample += z.size(0)
        if batch == 0 or batch == train_batch_num - 1:
            loss = loss.item()
            print(f"Batch: {batch:>3d}  loss: {loss:>12.7f}  [{current_sample:>5d}/{cal.Ns:>5d}]")


    avg_train_loss = total_train_loss / train_batch_num
    train_losses.append(avg_train_loss)

    # if avg_train_loss < cal.loss_min:
    #     break


    # 测试阶段
    # model.eval()
    # total_test_loss = 0
    # with torch.no_grad():
    #     for z,a,g,A,index in test_loader:
    #         loss = loss_fun(xi_fun, criterion, xi_fun_with_grad, index, a, g, ws, sml)
    #         total_test_loss += loss.item()
    #
    # avg_test_loss = total_test_loss / cal.batch_size
    # test_losses.append(avg_test_loss)

    # 每100个epoch打印一次损失
    if epoch % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.7f}')
    if (epoch + 1) % epoch_update_learn ==0:
        lr = adjust_learning_rate(optimizer, epoch, train_losses[-epoch_update_learn], train_losses[-1], lr)
    if avg_train_loss < cal.loss_min:
        break


np.savetxt("xi_fun_last", xi_fun(model, z, a,g,A).cpu().numpy(), delimiter=",")

# 绘制结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
#plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TrainingLoss')
plt.legend()



plt.tight_layout()
plt.show()