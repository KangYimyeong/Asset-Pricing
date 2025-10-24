import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square


# PCR
# sklearn未直接实现PCR模型，需先基于PCA生成X序列，然后执行线性模型，模型无法直接调参
def PCR(train_data, val_data, test_data):
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    test_X, test_y = DataLoader.data_split(test_data)

    n_components = [1, 2, 3, 5, 10, 20, 50]
    best_param = None
    R_square = - np.inf
    for n_component in n_components:
        pca = PCA(n_components=n_component)
        train_X_pca = pca.fit_transform(train_X)
        val_X_pca = pca.transform(val_X)

        model = LinearRegression()
        model.fit(train_X_pca, train_y)
        score = out_of_sample_R_square(model.predict(val_X_pca), val_y)
        print(f"PCR training: n_component={n_component}, R²={score}")
        if score > R_square:
            R_square = score
            best_param = n_component

    # 基于最优模型，评估测试集表现
    print(f"PCR best n_component: {best_param}")
    pca = PCA(n_components=best_param)
    train_X_pca = pca.fit_transform(train_X)
    test_X_pca = pca.transform(test_X)

    model = SGDRegressor(loss='huber')
    model.fit(train_X_pca, train_y)
    #     prediction_plot(model.predict(test_X_pca), test_y)
    print(f"PCR R²: {out_of_sample_R_square(model.predict(test_X_pca), test_y)}")
    return model.predict(test_X_pca), test_y, model
