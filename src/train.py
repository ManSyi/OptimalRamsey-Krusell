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

def train_model(model, cal):
    sml = Simulation(cal.Ns)
    sml.initial_sample()
    ws = WorkSpace(model, cal.batch_size)

    z = sml.z0
    a = sml.a0
    g = sml.g0
    A = sml.A0

    batch_size = cal.batch_size

    train_dataset = TensorDataset(z, a, g, A)

    train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator(device=cal.device_str))

    lr = cal.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    num_epochs = cal.num_epochs
    train_losses = []
    # test_losses = []

    criterion = nn.MSELoss()
    loss_fun = LossFunction()
    train_batch_num = len(train_loader)
    np.savetxt("xi_fun_start", xi_fun(model, z, a, g, A).cpu().numpy(), delimiter=",")

    train_dataset = TensorDataset(sml.z, sml.a, sml.g, sml.A)

    train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator(device=cal.device_str))

    train_batch_num = len(train_loader)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        current_sample = 0

        train_dataset = TensorDataset(sml.z, sml.a, sml.g, sml.A)

        train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator(device=cal.device_str))

        train_batch_num = len(train_loader)

        for batch, (z, a, g, A, index) in enumerate(train_loader):
            # ws.sample_size = z.shape[0]


            ws = WorkSpace(model, z.shape[0])
            ws.max_batch_size = z.shape[0]
            ws.z = z.clone()
            ws.A = A.clone()
            ws.a = a.clone()
            ws.g = g.clone()



            for t in range(cal.Nt):
                ws.dWj = (torch.normal(0, ws.dt, size=(ws.max_batch_size,), device=ws.device_str))
                ws.dWj_other = (torch.normal(0, ws.dt, size=(ws.max_batch_size, ws.Nj), device=ws.device_str))
                ws.dWk = (torch.normal(0, ws.dt, size=(ws.max_batch_size,), device=ws.device_str))
                ws.simulate_states_one_step(xi_fun, ws.dWj, ws.dWk, ws.dWj_other)

            sml.z[index] = ws.z.clone()
            sml.a[index] = ws.a.clone()
            sml.g[index, :, :] = ws.g.clone()
            sml.A[index] = ws.A.clone()
            # 前向传播

            loss = loss_fun(xi_fun, criterion, xi_fun_with_grad, ws, sml)

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
        if epoch % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.7f}')
        # if (epoch + 1) % 20 == 0:
        #     lr = lr * 0.1
        #     print(f"New learning rate = {lr}\n")
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # if (epoch + 1) % epoch_update_learn ==0:
        #     lr = adjust_learning_rate(optimizer, epoch, train_losses[-epoch_update_learn], train_losses[-1], lr)
        if avg_train_loss < cal.loss_min:
            break

    plt.figure(figsize=(12, 5))

    plt.plot(train_losses, label='Training Loss')
    # plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TrainingLoss')
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "model.pth")
    print("Saved Model to model.pth")



