import numpy as np
import pandas as pd
from itertools import product
from code.assess.MarginalAssociation import MarginalAssociation, MarginalAssociation_Interaction


def MarginalAssociation_Executor(trained_models, columns):
    print("MarginalAssociation_Executor started.")
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
    for model_name in trained_models:
        print(f"MarginalAssociation {model_name} started.")
        all_predictions = []
        if model_name in ["PCR", "GLM"]:
            print(f"MarginalAssociation {model_name} skipped.")
            continue
        if model_name == "OLS_3":
            characteristics_tmp = ["mvel1", "bm", "mom12m"]
            selected_columns = [col for col in columns if any(sub in col for sub in ["mvel1", "bm", "mom12m"])]
            for variable in characteristics_tmp:
                prediction = MarginalAssociation(trained_models[model_name], selected_columns, variable)
                all_predictions.append(pd.Series(prediction, name=f"{model_name}_{variable}"))
        elif model_name.startswith("NN"):
            for variable in characteristics:
                prediction = [MarginalAssociation(sub_model, columns, variable) for sub_model in
                              trained_models[model_name]]
                prediction = np.stack(prediction, axis=0).mean(axis=0)  # ensemble个模型取均值
                all_predictions.append(pd.Series(prediction, name=f"{model_name}_{variable}"))
        else:
            for variable in characteristics:
                prediction = MarginalAssociation(trained_models[model_name], columns, variable)
                all_predictions.append(pd.Series(prediction, name=f"{model_name}_{variable}"))
        MarginalAssociation_data = pd.concat(all_predictions, axis=1)
        MarginalAssociation_data["X"] = np.linspace(-1, 1, 100)
        print(f"MarginalAssociation {model_name} succeed.")
        MarginalAssociation_data.to_csv(f"/share/home/ymjiang/res/{model_name}_MarginalAssociation.csv", index=False)


def MarginalAssociation_Interaction_Excutor(trained_models, columns):
    print("MarginalAssociation_Interaction_Excutor started.")
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
    characteristics_product = list(product(characteristics, characteristics))
    values = [-1, -0.5, 0, 0.5, 1]
    for model_name in trained_models:  # 遍历模型
        print(f"MarginalAssociation_Interaction {model_name} started.")
        all_predictions = []
        if model_name in ["PCR", "GLM"]:
            print(f"MarginalAssociation_Interaction {model_name} skipped.")
            continue
        if model_name == "OLS_3":
            characteristics_product_tmp = product(["mvel1", "bm", "mom12m"], ["mvel1", "bm", "mom12m"])
            selected_columns = [col for col in columns if any(sub in col for sub in ["mvel1", "bm", "mom12m"])]
            for variable1, variable2 in characteristics_product_tmp:  # 遍历笛卡尔积组合
                if variable1 == variable2:
                    continue
                for value in values:  # 遍历参数取值
                    prediction = MarginalAssociation_Interaction(trained_models[model_name], selected_columns,
                                                                 variable1, variable2, value)
                    all_predictions.append(pd.Series(prediction, name=f"{model_name}_{variable1}_{variable2}={value}"))
        elif model_name.startswith("NN"):
            for variable1, variable2 in characteristics_product:  # 遍历笛卡尔积组合
                if variable1 == variable2:
                    continue
                for value in values:  # 遍历参数取值
                    prediction = [MarginalAssociation_Interaction(sub_model, columns, variable1, variable2, value) for
                                  sub_model in trained_models[model_name]]
                    prediction = np.stack(prediction, axis=0).mean(axis=0)
                    print(
                        f"MarginalAssociation_Interaction {model_name} {variable1}, {variable2} = {value}, prediction length: {len(prediction)}")
                    all_predictions.append(pd.Series(prediction, name=f"{model_name}_{variable1}_{variable2}={value}"))
        else:
            for variable1, variable2 in characteristics_product:  # 遍历笛卡尔积组合
                if variable1 == variable2:
                    continue
                for value in values:  # 遍历参数取值
                    prediction = MarginalAssociation_Interaction(trained_models[model_name], columns, variable1,
                                                                 variable2, value)
                    print(
                        f"MarginalAssociation_Interaction {model_name} {variable1}, {variable2} = {value}, prediction length: {len(prediction)}")
                    all_predictions.append(pd.Series(prediction, name=f"{model_name}_{variable1}_{variable2}={value}"))
        # 使用 pd.concat 一次性合并所有列
        MarginalAssociation_Interaction_data = pd.concat(all_predictions, axis=1)
        MarginalAssociation_Interaction_data["X"] = np.linspace(-1, 1, 100)
        print(f"MarginalAssociation_Interaction {model_name} succeed.")
        MarginalAssociation_Interaction_data.to_csv(
            f"/share/home/ymjiang/res/{model_name}_MarginalAssociation_Interaction.csv", index=False)
