import numpy as np
from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso


def GLM(train_data, val_data, test_data):
    # 切分数据
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    test_X, test_y = DataLoader.data_split(test_data)

    # 样条展开
    spline = SplineTransformer(degree=2, n_knots=3, include_bias=False)
    train_X_spline = spline.fit_transform(train_X)
    val_X_spline = spline.transform(val_X)
    test_X_spline = spline.transform(test_X)

    # 标准化特征
    scaler = StandardScaler()
    train_X_spline = scaler.fit_transform(train_X_spline)
    val_X_spline = scaler.transform(val_X_spline)
    test_X_spline = scaler.transform(test_X_spline)

    print(f"Transformed shapes: train={train_X_spline.shape}, "
          f"val={val_X_spline.shape}, test={test_X_spline.shape}")
    print(f"Check NaN: train={np.any(np.isnan(train_X_spline))}, "
          f"val={np.any(np.isnan(val_X_spline))}, test={np.any(np.isnan(test_X_spline))}")

    # 确定 group 索引
    n_groups = train_X.shape[1]  # 原始特征数
    n_basis = train_X_spline.shape[1] // n_groups  # 每个特征映射出的 basis 数
    groups = np.repeat(np.arange(n_groups), n_basis)

    print(f"Number of groups: {n_groups}, Basis per group: {n_basis}, Total features: {len(groups)}")

    # lambda 搜索
    lambdas = np.logspace(-4, -1, 15)  # [1e-4, 1e-1]
    best_param, best_r2 = None, -np.inf

    for lambda_ in lambdas:
        print("-" * 80)
        print(f"Training with lambda={lambda_:.6f}")

        model = GroupLasso(
            groups=groups,
            group_reg=lambda_,
            l1_reg=0,
            tol=1e-4,
            scale_reg="group_size",
            frobenius_lipschitz=True,
            subsampling_scheme=None,
            supress_warning=True
        )
        model.fit(train_X_spline, train_y)

        # 预测验证集
        val_pred = model.predict(val_X_spline)
        r2 = out_of_sample_R_square(val_pred, val_y)

        # 输出调试信息
        nonzero_total = np.sum(model.coef_ != 0)
        nonzero_groups = np.unique(groups[np.flatnonzero(model.coef_)]) if nonzero_total > 0 else []
        print(f"Validation R²: {r2:.6f}")
        print(f"First 5 predictions: {val_pred[:5]}")
        print(f"Nonzero coefficients: {nonzero_total}/{len(model.coef_)}")
        print(f"Nonzero groups: {len(nonzero_groups)} / {n_groups}")

        if r2 > best_r2:
            best_r2 = r2
            best_param = lambda_

    print("=" * 80)
    print(f"Best lambda: {best_param:.6f}, Best Validation R²: {best_r2:.6f}")

    # 用最优 lambda 训练最终模型
    model = GroupLasso(
        groups=groups,
        group_reg=best_param,
        l1_reg=0,
        tol=1e-4,
        scale_reg="group_size",
        frobenius_lipschitz=True,
        subsampling_scheme=None,
        supress_warning=True
    )
    model.fit(train_X_spline, train_y)

    # 测试集结果
    test_pred = model.predict(test_X_spline)
    test_r2 = out_of_sample_R_square(test_pred, test_y)

    nonzero_total = np.sum(model.coef_ != 0)
    nonzero_groups = np.unique(groups[np.flatnonzero(model.coef_)]) if nonzero_total > 0 else []

    print("=" * 80)
    print(f"Final Test R²: {test_r2:.6f}")
    print(f"Test predictions (first 5): {test_pred[:5]}")
    print(f"Final nonzero coefficients: {nonzero_total}/{len(model.coef_)}")
    print(f"Final nonzero groups: {len(nonzero_groups)} / {n_groups}")

    return test_pred, test_y, model