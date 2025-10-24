import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from DataLoader import DataLoader


# out-of-sample R²
def out_of_sample_R_square(r_pred, r_true):
    """
    计算样本外R2
    r_pred: np.array, 验证集模型预测超额收益率
    r_true: np.array, 验证集实际超额收益率
    """
    if type(r_pred) == list:
        r_pred = np.array(r_pred)
    if type(r_true) == list:
        r_true = np.array(r_true)
    return 1 - (np.sum((r_true - r_pred)**2) / np.sum(r_true**2))


# Parameter Adjustment gridSearch
def gridSearch(model_builder, train_data, val_data, params=None):
    """
    model_type: 模型类型函数
    params: 参数字典
    train_data: 训练集
    val_data: 验证集
    """
    if params is None:
        params = {}
    # 1. 准备数据
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    # 2. 准备模型参数
    keys = list(params.keys()) # 参数名
    param_combination = list(product(*params.values())) # 将各个参数做笛卡尔积(参数格式[[...], [...], [...]])
    # 3. 全局变量
    best_estimator, best_param = None, None # 最佳模型/最佳参数组合
    R_square = - np.inf
    # 4. 迭代参数组合
    for param in param_combination:
        print(f"training: {dict(zip(keys, param))}", end=", ") # {param1: value1, param2, value2}, 然后解包
        model = model_builder(**dict(zip(keys, param)))
        model.fit(train_X, train_y)
        score = out_of_sample_R_square(model.predict(val_X), val_y)
        print(f"R²: {score}")
        if score > R_square:
            best_estimator = model
            best_param = dict(zip(keys, param))
            R_square = score
    print(f"best_param: {best_param}, best_R²: {R_square}")
    return {"best_estimator": best_estimator, "best_param": best_param, "best_R²": R_square}


# Plot predictions
def prediction_plot(y_pred, y_true):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    plt.tight_layout()
    ax[0, 0].plot(range(0, len(y_true)), y_true, label="true value")
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].plot(range(0, len(y_pred)), y_pred, label="model prediction", alpha=0.7)
    ax[0, 1].grid()
    ax[0, 1].legend()

    ax[1, 0].hist(y_true, label="true value")
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].hist(y_pred, label="model prediction", alpha=0.7)
    ax[1, 1].grid()
    ax[1, 1].legend()
    plt.show()


