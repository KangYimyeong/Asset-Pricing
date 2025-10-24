import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from group_lasso import GroupLasso
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import multiprocessing as mp
from DataLoader import DataLoader
from code.utils import out_of_sample_R_square, gridSearch, prediction_plot
from code.model.OLS import OLS
from code.model.OLS import OLS_3
from code.model.GLM import GLM
from code.model.ENet import ENet
from code.model.PCR import PCR
from code.model.GRBT import GBRT
from code.model.PLS import PLS
from code.model.RandomForest import RF
from code.model.NuralNetwork import NN, NN_def
from code.model.LightGBM import LightGBM
from code.model.XGBoost import XGBoost
from code.model.CatBoost import CatBoost
from code.assess.executor import MarginalAssociation_Executor, MarginalAssociation_Interaction_Excutor
from code.model.LSTM import LSTM
from code.model.Transformer import Transformer
from code.assess.DMTest import cross_sectional_averaged_DM


def train_model(name, train_data, val_data, test_data):
    """统一训练接口，返回模型预测结果"""
    if name == "OLS":
        ret = OLS(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "OLS_3":
        ret = OLS_3(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "PLS":
        ret = PLS(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "PCR":
        ret = PCR(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "ENet":
        ret = ENet(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "GLM":
        ret = GLM(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "RF":
        ret = RF(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "GBRT":
        ret = GBRT(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "LightGBM":
        ret = LightGBM(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "XGBoost":
        ret = XGBoost(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "CatBoost":
        ret = CatBoost(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "LSTM":
        ret = LSTM(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name == "Transformer":
        ret = Transformer(train_data, val_data, test_data)
        return name, ret[0], ret[1], ret[2]
    elif name.startswith("NN"):
        layer_dict = {
            "NN1": [32],
            "NN2": [32, 16],
            "NN3": [32, 16, 8],
            "NN4": [32, 16, 8, 4],
            "NN5": [32, 16, 8, 4, 2]
        }
        ret = NN(model_class=NN_def, model_kwargs={"layer_sizes": layer_dict[name]}, train_data=train_data, val_data=val_data, test_data=test_data)
        torch.cuda.empty_cache()
        return name, ret[0], ret[1], ret[2]
    else:
        raise ValueError(f"Unknown model {name}")


def main():
    parser = argparse.ArgumentParser(description="Asset Pricing via ML - Rolling Window Training")

    # 时间窗口参数
    parser.add_argument("--train_start", type=int, default=1957, help="训练起始年份")
    parser.add_argument("--train_end", type=int, default=1974, help="训练结束年份")
    parser.add_argument("--val_end", type=int, default=1986, help="验证集结束年份")
    parser.add_argument("--test_end", type=int, default=2016, help="测试集结束年份")

    # 模型选择，可以用逗号分隔传多个
    parser.add_argument(
        "--models",
        type=str,
        default="OLS,OLS_3,PLS,PCR,ENet,GLM,RF,GBRT,NN1,NN2,NN3,NN4,NN5",
        help="选择要训练的模型，多个模型用逗号分隔"
    )

    # 最大进程数
    parser.add_argument("--max_workers", type=int, default=3, help="最大并行进程数")

    # 输出目录
    parser.add_argument("--output_dir", type=str, default="/share/home/ymjiang/res", help="结果保存目录")

    args = parser.parse_args()

    # 解析模型列表
    model_names = [m.strip() for m in args.models.split(",")]
    print("Model List: ", model_names)
    # 数据加载器
    dl1 = DataLoader(
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        test_end=args.test_end
    )

    idx = 0
    trained_models = {}

    while True:
        idx += 1
        print(f"Round-{idx} started.")
        train_data, val_data, test_data = dl1.load_data()
        if train_data is None and val_data is None and test_data is None:
            break
        
        res_data_all = pd.DataFrame()
        res_data = test_data[["permno", "DATE", "EXRET"]].copy()
        print(f"train_data: {train_data.shape}, val_data: {val_data.shape}, test_data: {test_data.shape}")

        # 多进程训练
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(train_model, name, train_data, val_data, test_data): name for name in model_names}
            for future in as_completed(futures):
                model_name, pred, test_data, model = future.result()
                res_data[model_name] = pred
                trained_models[model_name] = model
                # test_data_map[model_name] = test_data

        res_data_all = pd.concat([res_data_all, res_data], axis=0)
        res_data_all.to_csv(f"{args.output_dir}/{dl1.train_end_now + dl1.val_len + 1}_{idx}.csv", index=False)

    columns = DataLoader.data_split(pd.read_csv(f"/share/home/ymjiang/data/{args.test_end}.csv"))[0].columns
    MarginalAssociation_Executor(trained_models, columns)
    MarginalAssociation_Interaction_Excutor(trained_models, columns)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()