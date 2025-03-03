import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
def train_model(model, dataset, args, device, model_dir='models', loss_function=nn.MSELoss()):
    os.makedirs(model_dir, exist_ok=True)
    best_model_filename = os.path.join(model_dir, 'best_model.pth')
    best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
    # 损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(torch.tensor(dataset.trainx).float(), torch.tensor(dataset.trainy).float())
    test_dataset = TensorDataset(torch.tensor(dataset.vaildx).float(), torch.tensor(dataset.vaildy).float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 开始时间
    AllTrainEpoch_start_time = time.time()
    train_losses, test_losses = [], []
    pbar = tqdm(range(args.epochs), desc='Training...')
    for epoch in pbar:
        start_time = time.time()
        model.train()
        train_loss_total = 0  # 用于累加每个epoch的训练损失
        num_batches = 0
        for trainx_batch, trainy_batch in train_loader:
            trainx_batch, trainy_batch = trainx_batch.to(device), trainy_batch.to(device)
            optimizer.zero_grad()
            model_output = model(trainx_batch)
            model_output = model_output.squeeze()
            train_loss = loss_function(model_output, trainy_batch)
            train_loss.backward()
            optimizer.step()
            train_loss_total += train_loss.item()
            num_batches += 1
        train_loss_avg = train_loss_total / num_batches
        train_losses.append(train_loss_avg)  # 记录每个epoch的平均训练损失

        # 模型测试，加入耐心机制，防止过拟合
        model.eval()
        test_loss_total = 0
        num_test_batches = 0
        with torch.no_grad():
            for testx_batch, testy_batch in test_loader:
                testx_batch, testy_batch = testx_batch.to(device), testy_batch.to(device)
                test_output = model(testx_batch)
                test_output = test_output.squeeze()
                test_loss = loss_function(test_output, testy_batch)
                test_loss_total += test_loss.item()
                num_test_batches += 1
        test_loss_avg = test_loss_total / num_test_batches  # 计算平均测试损失
        test_losses.append(test_loss_avg)
        end_time = time.time()

        # 如果当前测试损失低于之前记录的最佳测试损失，则保存模型
        if test_loss_avg < best_test_loss:
            i = 0
            best_test_loss = test_loss_avg
            torch.save(model.state_dict(), best_model_filename)
            pbar.set_description(f"轮次: {epoch}, 训练损失: {train_loss_avg:.10f}, 测试损失: {test_loss_avg:.10f}, 训练时间:{(end_time - start_time):.3f}, 早停次数:{i}")
        else:
            i += 1  # 说明损失函数已经多少次训练未降低
            pbar.set_description(f"轮次: {epoch}, 训练损失: {train_loss_avg:.10f}, 测试损失: {test_loss_avg:.10f}, 训练时间:{(end_time - start_time):.3f}, 早停次数:{i}")
            if i == args.patience:
                with open(os.path.join(model_dir, f'训练轮次为{epoch}.txt'), 'w') as file:
                    pass
                print("触发早停机制!")
                break
    # 结束时间
    AllTrainEpoch_end_time = time.time()
    print(f'总训练时间：{AllTrainEpoch_end_time - AllTrainEpoch_start_time}秒')
    # 转化成txt保存
    with open(os.path.join(model_dir, f'new训练时间：{AllTrainEpoch_end_time - AllTrainEpoch_start_time}秒.txt'), 'w') as file:
        pass

    # 可视化，保存损失函数图像
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Vaild Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Vaild Loss Over Epochs')
    plt.legend()
    loss_name = os.path.join(model_dir, 'loss.png')
    plt.savefig(loss_name)
    plt.close()

    # 保存损失函数数据
    # 将列表转换为NumPy数组
    train_losses_array = np.array(train_losses)
    test_losses_array = np.array(test_losses)
    # 保存NumPy数组
    train_losses_path = os.path.join(model_dir, 'train_losses.npy')
    test_losses_path = os.path.join(model_dir, 'test_losses.npy')
    np.save(train_losses_path, train_losses_array)
    np.save(test_losses_path, test_losses_array)