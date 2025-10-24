from code.DataLoader import DataLoader
from code.utils import out_of_sample_R_square
from sklearn.linear_model import SGDRegressor


# OLS+H (ordinary least squares with huber loss function)
def OLS(train_data, val_data, test_data):
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    test_X, test_y = DataLoader.data_split(test_data)

    model = SGDRegressor(loss='huber')
    model.fit(train_X, train_y)
    print(f"OLS R²: {out_of_sample_R_square(model.predict(test_X), test_y)}")
    return model.predict(test_X), test_y, model


# OLS-3+H
# (Ordinary Least Squares on size/book-to-market/momentum with huber loss function)
def OLS_3(train_data, val_data, test_data):
    train_X, train_y = DataLoader.data_split(train_data)
    val_X, val_y = DataLoader.data_split(val_data)
    test_X, test_y = DataLoader.data_split(test_data)

    selected_columns = [col for col in train_X.columns if any(sub in col for sub in ["mvel1", "bm", "mom12m"])]
    train_X = train_X[selected_columns]
    val_X = val_X[selected_columns]
    test_X = test_X[selected_columns]
    model = SGDRegressor(loss='huber')
    model.fit(train_X, train_y)
    #     prediction_plot(model.predict(test_X), test_y)
    print(f"OLS-3 R²: {out_of_sample_R_square(model.predict(test_X), test_y)}")
    return model.predict(test_X), test_y, model