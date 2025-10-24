```text
Project/  
|- code/  
|    |- __init__.py
|    |- utils.py
|    |- DataPreparation.py
|    |- DataLoader.py
|    |- ModelPrediction.py
|    |- assess/
|    |    |- DMTest.py
|    |    |- executor.py
|    |    |- MarginalAssociation.py
|    |- model/
|         |- OLS.py
|         |- PLS.py
|         |- PCR.py
|         |- GLM.py
|         |- ENet.py
|         |- RandomForest.py
|         |- GRBT.py
|         |- LightGBM.py
|         |- XGBoost.py
|         |- CatBoost.py
|         |- NuralNetWork.py
|         |- LSTM.py
|         |- Transformer.py
|- raw/
    |- MacroData.csv
    |- StockReturn.parquet
    |- FeatureData.csv
```
- The datasets `FeatureData.csv` and `StockReturn.csv` can be downloaded from Kaggle: https://www.kaggle.com/datasets/kyouichimei/dataset-for-asset-pricing
