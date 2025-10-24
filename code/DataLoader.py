import numpy as np
import pandas as pd


class DataLoader():
    def __init__(self, train_start=1957, train_end=1974, val_end=1986, test_end=2016):
        """ 逐次返回训练/验证/测试集数据
        train_start: 训练集起始年份
        train_end: 训练集结束年份，训练集[train_start: train_end]
        val_end: 验证集结束年份，验证集[train_end+1: val_end]
        test_end: 测试集结束年份，测试集[val_end+1]
        """
        # def __init__(self, train_start=1957, train_end=1958, val_end=1960, test_end=1963):
        self.train_data, self.val_data, self.test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.train_start = train_start
        self.train_end, self.val_end, self.test_end = train_end, val_end, test_end
        self.val_len = val_end - train_end  # 验证集时间长度(默认12年)
        self.train_end_now = train_end  # 当前训练集结束时间

    @staticmethod
    def date_format(df):
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
        return df

    def load_data(self):
        """
        首次直接读取初始训练集、验证集及测试集，然后逐年更新。
        return: train_data, val_data, test_data (若无新数据可更新, 返回 None, None, None)
        """
        if self.train_end_now == self.train_end:
            # 1.准备初始训练集
            print(f"初始化训练集: ", end="")
            for i in range(self.train_start, self.train_end + 1):
                # 直接读取1957–1974作为初始训练集
                print(f"{i}", end=", ")
                self.train_data = pd.concat(
                    [self.train_data, self.date_format(pd.read_csv(f"/share/home/ymjiang/data/{i}.csv"))], axis=0)
            # 2.准备初始验证集
            print(f"初始化验证集: ", end="")
            for i in range(self.train_end + 1, self.val_end + 1):
                # 直接读取1975-1986作为初始验证集
                print(f"{i}", end=", ")
                self.val_data = pd.concat(
                    [self.val_data, self.date_format(pd.read_csv(f"/share/home/ymjiang/data/{i}.csv"))], axis=0)
            # 3.准备初始测试集
            print(f"初始化验证集: {self.val_end + 1}")
            self.test_data = self.date_format(pd.read_csv(f"/share/home/ymjiang/data/{self.val_end + 1}.csv"))

            self.train_end_now += 1
            return self.train_data, self.val_data, self.test_data
        else:
            if self.train_end_now + self.val_len < self.test_end:
                # 1. 训练集逐年向后延展
                print(f"扩展训练集: {self.train_end_now}", end=", ")
                self.train_data = pd.concat([self.train_data, self.date_format(
                    pd.read_csv(f"/share/home/ymjiang/data/{self.train_end_now}.csv"))], axis=0)
                # 2. 验证集向后roll forward
                print(f"删除验证集: {self.train_end_now}, 扩展验证集: {self.train_end_now + self.val_len}", end=", ")
                self.val_data = pd.concat([self.val_data, self.date_format(
                    pd.read_csv(f"/share/home/ymjiang/data/{self.train_end_now + self.val_len}.csv"))], axis=0)
                self.val_data = self.val_data[self.val_data["DATE"] >= pd.to_datetime(f"{self.train_end_now + 1}-01-01",
                                                                                      format="%Y-%m-%d")]  # 过滤掉前一年
                # 3. 测试集固定取1年
                print(f"更新测试集: {self.train_end_now + self.val_len + 1}")
                self.test_data = self.date_format(
                    pd.read_csv(f"/share/home/ymjiang/data/{self.train_end_now + self.val_len + 1}.csv"))

                self.train_end_now += 1
                return self.train_data, self.val_data, self.test_data
            else:
                print("数据集生成完成")
                return None, None, None

    @staticmethod
    def data_split(data):
        """
        用于将数据集拆分为X, y两部分
        """
        return data.drop(["permno", "DATE", "RET", "EXRET"], axis=1).astype('float32'), data["EXRET"].astype('float32')
