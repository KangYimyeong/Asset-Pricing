import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
import copy


# ### 5.9 NN1 - NN5
class NN_def(nn.Module):
    """
    实现自定义层数和隐藏层大小的神经网络
    """

    def __init__(self, input_size=920, layer_sizes=[32, 16, 8, 4, 2], output_size=1):
        super().__init__()

        layers = []

        for i in range(0, len(layer_sizes)):
            if i == 0:
                linear = nn.Linear(input_size, layer_sizes[i])
            else:
                linear = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # He 初始化（Kaiming 初始化）
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            layers.append(linear)
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.ReLU())

        output_layer = nn.Linear(layer_sizes[-1], output_size)
        nn.init.uniform_(output_layer.weight, -0.1, 0.1)  # 小幅初始化
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)  # 设备兼容
        return self.model(x).squeeze()

    # 与其他模型统一接口
    def predict(self, x):
        if type(x) == pd.DataFrame:
            x = torch.tensor(x.values, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)  # 设备兼容
        return self.model(x).squeeze().detach().cpu().numpy()


def NN(train_data, val_data, test_data, model_class=NN_def, model_kwargs=None,
       learning_rate=[0.001, 0.01], batch_size=10000, epochs=100, patience=5, ensemble=10):
    """
    神经网络训练框架
    model_class: 模型类型
    model_kwargs: 模型参数
    train_data: 训练集数据
    val_data: 验证集数据
    test_data: 测试集数据
    learning_rate: 学习率
    batch_size: 批大小
    epochs: 迭代轮数
    patience: 早停轮数
    ensemble: 集成轮数
    """
    # 1. 准备数据集
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    test_X, test_y = DataLoader.data_split(test_data)

    # 将dataframe转换为tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_X, train_y = torch.tensor(train_X.values, dtype=torch.float32).to(device), torch.tensor(train_y.values,
                                                                                                  dtype=torch.float32).to(
        device)
    val_X, val_y = torch.tensor(val_X.values, dtype=torch.float32).to(device), torch.tensor(val_y.values,
                                                                                            dtype=torch.float32).to(
        device)
    test_X, test_y = torch.tensor(test_X.values, dtype=torch.float32).to(device), torch.tensor(test_y.values,
                                                                                               dtype=torch.float32).to(
        device)
    # 数据分batch
    dataset = torch.utils.data.TensorDataset(train_X, train_y)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

    model_state_group = []  # 记录每种参数组合下10个模型，然后根据总体表现筛选出最优组
    for lr in learning_rate:
        best_model_state = [None for i in range(ensemble)]  # 最优模型参数
        for e in range(0, ensemble):
            seed = e
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 2. 准备模型
            model = model_class(**model_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.L1Loss()

            best_loss = np.inf
            no_improve_epochs = 0

            for epoch in range(epochs):
                # 3. 训练模型
                model.train()
                for i, (X, y) in enumerate(data_loader):  # y: torch.Size([10000])
                    optimizer.zero_grad()
                    train_pred = model(X)
                    train_loss = criterion(train_pred, y)
                    train_loss.backward()
                    optimizer.step()
                # 4. 验证模型
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_X)
                    val_loss = criterion(val_pred, val_y)
                    print(f"Epoch {epoch + 1}, Val Loss: {val_loss.item():.4f}")
                # 5. early stopping判断
                if val_loss < best_loss:
                    best_loss = val_loss.item()
                    best_model_state[e] = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs > patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
        model_state_group.append(best_model_state)

    # 6. 筛选最优组（每组有ensemble个模型）
    best_loss = np.inf
    best_model_state_group = None
    for i in range(0, len(learning_rate)):
        val_pred_list = []
        for j in range(0, len(model_state_group[i])):
            model = model_class(**model_kwargs).to(device)
            model.load_state_dict(model_state_group[i][j])
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X)
                val_pred_list.append(val_pred)
        loss = criterion(torch.stack(val_pred_list).mean(dim=0), val_y)
        if loss < best_loss:
            best_loss = loss
            best_model_state_group = model_state_group[i]

    # 6. 使用最优组预测
    if best_model_state_group:
        test_pred_list = []
        model_list = []
        for i in range(0, ensemble):
            model = model_class(**model_kwargs).to(device)
            model.load_state_dict(best_model_state_group[i])
            model.eval()
            with torch.no_grad():
                test_pred = model(test_X).squeeze()
            model_list.append(model)
            test_pred_list.append(test_pred)

        test_y = test_y.squeeze()
        print(f"NN Test Loss (L1): {criterion(torch.stack(test_pred_list).mean(dim=0), test_y).item():.4f}")
        # tensor→numpy
        test_pred_mean = torch.stack(test_pred_list).mean(dim=0).detach().cpu().numpy()
        test_y = test_y.detach().cpu().numpy()
        print(f"NN Test Loss (R²): {out_of_sample_R_square(test_pred_mean, test_y)}")
        return test_pred_mean, test_y, model_list
    else:
        raise RuntimeError("No best model because the model was not updated.")