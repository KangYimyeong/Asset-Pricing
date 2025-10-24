import numpy as np
import pandas as pd


def MarginalAssociation(model, columns, variable, model_kwargs=None):
    """
    model: 训练完成的模型
    test_data: DataFrame，模型测试集
    variable：str, 要研究的变量，取值为[-1, 1]
    model_kwargs: dict, 赋值给model_class的属性
    """
    # 为保证数据格式与model训练时的一致，直接取test_X的columns
    df_data = pd.DataFrame(0, index=np.arange(100), columns=columns)
    # 设置研究变量的取值范围为[-1, 1]
    df_data[f"cons_{variable}"] = list(np.linspace(-1, 1, 100))
    # 其他变量直接置为0
    df_data.fillna(0, inplace=True)
    prediction = model.predict(df_data)
    return prediction


def MarginalAssociation_Interaction(model, columns, variable1, variable2, value, model_kwargs=None):
    """
    model: 训练完成的模型
    test_data: DataFrame，模型测试集
    variable1：str, 要研究的变量名，取值为[-1, 1]
    variable2: str, 要研究的变量名，取值为定值
    value: float/int/double, variable2的取值
    model_kwargs: dict, 赋值给model_class的属性
    """
    # 为保证数据格式与model训练时的一致，直接取test_X的columns
    df_data = pd.DataFrame(0, index=np.arange(100), columns=columns)
    # 设置研究变量的取值范围为[-1, 1]
    df_data[f"cons_{variable1}"] = list(np.linspace(-1, 1, 100))
    df_data[f"cons_{variable2}"] = value
    # 其他变量直接置为0
    df_data.fillna(0, inplace=True)
    prediction = model.predict(df_data)
    return prediction


