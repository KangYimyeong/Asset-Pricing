import numpy as np
import pandas as pd

def cross_sectional_averaged_DM(pred_df, key1, key2, key_true="EXRET"):
    """
    构造统计量比较模型优劣
    pred_df: DataFrame类型, 包含"DATE", "permno", 各机器学习算法预测值
    key1: 模型1 col名
    key2: 模型2 col名
    key_true: 真实超额收益col名
    """
    pred_df = pred_df.copy()
    pred_df["e1"] = pred_df[key1] - pred_df[key_true]
    pred_df["e2"] = pred_df[key2] - pred_df[key_true]
    pred_df["diff"] = pred_df["e1"] ** 2 - pred_df["e2"] ** 2

    series = pred_df.groupby("DATE")["diff"].mean().reset_index()
    DM = series["diff"].mean() / series["diff"].std()
    return DM