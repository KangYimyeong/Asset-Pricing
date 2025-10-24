import pandas as pd
import numpy as np
from itertools import product

# 1. 因子数据处理
# 缺失值填充与排序映射
# 94个变量（注意tableA.6中写的虽然是grCAPX，但是实际是grcapx）
characteristics = [
    "absacc", "acc", "aeavol", "age", "agr", "baspread", "beta", 
    "betasq", "bm", "bm_ia", "cash", "cashdebt", "cashpr", "cfp",
    "cfp_ia", "chatoia", "chcsho", "chempia", "chinv", "chmom", 
    "chpmia", "chtx", "cinvest", "convind", "currat", "depr", "divi",
    "divo", "dolvol", "dy", "ear", "egr", "ep", "gma", "grcapx", 
    "grltnoa", "herf", "hire", "idiovol", "ill", "indmom", "invest",
    "lev", "lgr", "maxret", "mom12m", "mom1m", "mom36m", "mom6m", "ms", 
    "mvel1", "mve_ia", "nincr", "operprof", "orgcap", "pchcapx_ia", "pchcurrat",
    "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt", "pchsale_pchrect",
    "pchsale_pchxsga", "pchsaleinv", "pctacc", "pricedelay", "ps", "quick",
    "rd", "rd_mve", "rd_sale", "realestate", "retvol", "roaq", "roavol", "roeq", 
    "roic", "rsup", "salecash", "saleinv", "salerec", "secured", "securedind", 
    "sgr", "sin", "sp", "std_dolvol", "std_turn", "stdacc", "stdcf", "tang", "tb", 
    "turn", "zerotrade"
]


df = pd.read_csv("/share/home/ymjiang/data/FeatureData.csv")
df = df[(df["DATE"]>=19570000) & (df["DATE"]<=20170000)]

# 1.1 缺失值填充
# 按月用横截面中位数填充
df[characteristics] = df.groupby("DATE")[characteristics].transform(lambda x: x.fillna(x.median()))
# 将剩余缺失值一律置为 0
df[characteristics] = df[characteristics].fillna(0)
pd.DataFrame({
    'unique': df.nunique() == len(df),
    'cardinality': df.nunique(),
    'with_null': df.isna().any(),
    'null_pct': round(100 * (df.isnull().sum() / len(df)), 2),
    '0_pct': round(100 * ((df==0).sum() / len(df)), 2),
    '1st_row': df.iloc[0],
    'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
    'last_row': df.iloc[-1],
    'dtype': df.dtypes
})


# 1.2 排序映射
df[characteristics] = df.groupby("DATE")[characteristics].transform(lambda x: x.rank(pct=True)) * 2 - 1
df.describe()

# 2. 合并收益率数据
# 同时剔除收益率缺失样本
# 2.1 收益率数据处理与合并
df_return = pd.read_csv("/share/home/ymjiang/data/StockReturn.csv")
df_return.rename({"PERMNO": "permno", "date": "DATE"}, axis=1, inplace=True)
df_return["DATE"] = df_return["DATE"].astype(int)
df_return['RET'] = pd.to_numeric(df_return['RET'], errors='coerce') # RET有出现'B', 'C'
df = pd.merge(left=df, right=df_return, how="left", on=["permno", "DATE"])
# 2.2 剔除无收益率的样本
# 此前已将缺失值填充为0，此时的缺失值必然是因为收益率而导致的缺失
df.dropna(how="any", inplace=True)

# 3. 宏观字段处理与合并
df_macro = pd.read_csv("/share/home/ymjiang/data/MacroData.csv")
df_macro = df_macro[(df_macro["yyyymm"] >= 195701) & df_macro["yyyymm"] <= 201700]
df_macro["macro_dp"] = df_macro["D12"]
df_macro["macro_ep"] = df_macro["E12"]
df_macro["macro_bm"] = df_macro["b/m"]
df_macro["macro_ntis"] = df_macro["ntis"]
df_macro["macro_tbl"] = df_macro["tbl"]
df_macro["macro_tms"] = df_macro["lty"] - df_macro["tbl"]
df_macro["macro_dfy"] = df_macro["BAA"] - df_macro["AAA"]
df_macro["macro_svar"] = df_macro["svar"]
df_macro = df_macro[["yyyymm", "macro_dp", "macro_ep", "macro_bm", "macro_ntis", "macro_tbl", "macro_tms", "macro_dfy", "macro_svar"]]
df["yyyymm"] = (df["DATE"]/100).astype(int) # 单独提取出年月用于合并
df = pd.merge(left=df, right=df_macro, how="left", on="yyyymm")
# 计算超额收益率
df["EXRET"] = df["RET"] - df["macro_tbl"] # 超额收益率

# 4. 行业虚拟变量处理与合并
industry_dummies = pd.get_dummies(df['sic2'], prefix='industry', dummy_na=False)
if industry_dummies.shape[0] == df.shape[0]:
    df = pd.concat([df, industry_dummies], axis=1)

# 5. 计算交叉属性前保存全量数据
df.drop(columns=["sic2", "yyyymm"], inplace=True)
df.to_csv("/share/home/ymjiang/data/FeatureData_Revised.csv", index=False)

# 6. 构造交叉特征
# 由于原数据大小极大，在此选择逐年拆分后分别计算交叉特征并存储
# df = pd.read_csv(r"F:/Paper_ML/datashare_revision.csv")
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
df["Year"] = df["DATE"].dt.year
df = df[(df["Year"] >= 1957) & (df["Year"] <= 2016)]
df["cons"] = 1 # 常数变量
df_groups = df.groupby("Year")
for year, sub_df in df_groups:
    print(year, end=", ")
    basic = ["permno", "DATE", "RET", "EXRET"] # 基础变量(编号, 日期, 收益率, 超额收益率)
    industry_ = [ele for ele in df.columns if ele.startswith("industry_")] # 行业虚拟变量
    macro_ = [ele for ele in df.columns if ele.startswith("macro_")] # 宏观变量
    macro_.append("cons")
    features_product = list(product(macro_, characteristics)) # 宏观变量与一般变量的笛卡尔积
    cross_columns = [(sub_df[col1] * sub_df[col2]).values for col1, col2 in features_product]
    cross_columns = pd.DataFrame(cross_columns, index=[f"{col1}_{col2}" for col1, col2 in features_product]).T
    cross_columns.index = sub_df.index # 需保证index对齐
    
    sub_df = pd.concat([sub_df, cross_columns], axis=1)
    sub_df = sub_df[basic + industry_ + [f"{col1}_{col2}" for col1, col2 in features_product]]
    sub_df.to_csv(rf"/share/home/ymjiang/data/{year}.csv", index=False)
