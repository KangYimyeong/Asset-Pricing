import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
import copy


class LSTM_def(nn.Module):
    """
    自定义 LSTM 网络，支持多层 LSTM + 全连接输出
    """

    def __init__(self, input_size=920, hidden_size=64, num_layers=2, output_size=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        # 权重初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size], 对于常规 tabular 可以设 seq_len=1
        x = x.to(next(self.parameters()).device)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加 seq_len 维度
        out, _ = self.lstm(x)  # out: [batch_size, seq_len, hidden_size]
        out = out[:, -1, :]  # 取最后一个时间步
        return self.fc(out).squeeze()

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.forward(x).detach().cpu().numpy()


def LSTM(train_data, val_data, test_data, model_class=LSTM_def, model_kwargs=None,
         learning_rate=[0.001, 0.01], batch_size=10000, epochs=100, patience=5, ensemble=5):
    """
    LSTM训练框架，接口与 NN 函数一致
    """
    # 1. 准备数据集
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    test_X, test_y = DataLoader.data_split(test_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_X = torch.tensor(train_X.values, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y.values, dtype=torch.float32).to(device)
    val_X = torch.tensor(val_X.values, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y.values, dtype=torch.float32).to(device)
    test_X = torch.tensor(test_X.values, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y.values, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(train_X, train_y)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

    model_state_group = []
    for lr in learning_rate:
        best_model_state = [None for _ in range(ensemble)]
        for e in range(ensemble):
            seed = e
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = model_class(**model_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.L1Loss()

            best_loss = np.inf
            no_improve_epochs = 0

            for epoch in range(epochs):
                model.train()
                for X_batch, y_batch in data_loader:
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(val_X)
                    val_loss = criterion(val_pred, val_y)
                print(f"LR {lr}, Seed {seed}, Epoch {epoch + 1}, Val Loss: {val_loss.item():.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss.item()
                    best_model_state[e] = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs > patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        model_state_group.append(best_model_state)

    # 选择最优组
    best_loss = np.inf
    best_model_state_group = None
    criterion = nn.L1Loss()
    for i in range(len(learning_rate)):
        val_pred_list = []
        for j in range(len(model_state_group[i])):
            model = model_class(**model_kwargs).to(device)
            model.load_state_dict(model_state_group[i][j])
            model.eval()
            with torch.no_grad():
                val_pred_list.append(model(val_X))
        loss = criterion(torch.stack(val_pred_list).mean(dim=0), val_y)
        if loss < best_loss:
            best_loss = loss
            best_model_state_group = model_state_group[i]

    # 使用最优组预测
    test_pred_list = []
    model_list = []
    for state in best_model_state_group:
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            test_pred_list.append(model(test_X))
        model_list.append(model)

    test_pred_mean = torch.stack(test_pred_list).mean(dim=0).detach().cpu().numpy()
    test_y_np = test_y.detach().cpu().numpy()
    print(f"LSTM Test Loss (L1): {criterion(torch.stack(test_pred_list).mean(dim=0), test_y).item():.4f}")
    print(f"LSTM Test R²: {out_of_sample_R_square(test_pred_mean, test_y_np)}")

    return test_pred_mean, test_y_np, model_list
