#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:Lijiacai
Email:1050518702@qq.com
===========================================
CopyRight@JackLee.com
===========================================
"""

import os
import sys
import json
import pandas
import numpy
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import ensemble
import matplotlib.pyplot as plt


class RandomeTree1():
    def read_data(self, filepath_or_buffer="./员工离职预测训练赛/pfm_train.csv", sep=","):
        data = pandas.read_table(filepath_or_buffer=filepath_or_buffer, sep=sep)
        return data

    def pre_data(self, data):
        dummies_col = ["BusinessTravel", "Department", "EducationField", "JobRole", "Gender", "MaritalStatus", "Over18",
                       "OverTime"]
        dum_l = []
        for i in dummies_col:
            dum_l.append(pandas.get_dummies(data[i]))
        l = [data] + dum_l
        data = pandas.concat(l, axis=1)
        data.drop(dummies_col, axis=1, inplace=True)
        new_data = data.loc[:, ["Age", "MonthlyIncome", "PercentSalaryHike",
                                #                         "PerformanceRating",
                                #                         "RelationshipSatisfaction",
                                #                         "StockOptionLevel",
                                #                         "TotalWorkingYears",
                                #                         "TrainingTimesLastYear",
                                #                         "WorkLifeBalance",
                                #                         "YearsAtCompany",
                                #                         "YearsInCurrentRole"
                                ]]
        new_data = (new_data - new_data.mean()) / new_data.std()
        # new_data = (new_data - new_data.min()) / (new_data.max() - new_data.min())
        data.drop(["Age", "MonthlyIncome", "EmployeeNumber", "PercentSalaryHike",
                   # "PerformanceRating",
                   # "RelationshipSatisfaction",
                   # "StandardHours",
                   # "StockOptionLevel",
                   # "TotalWorkingYears",
                   # "TrainingTimesLastYear",
                   # "WorkLifeBalance",
                   # "YearsAtCompany",
                   # "YearsInCurrentRole"
                   ], axis=1, inplace=True)
        data = pandas.concat([data, new_data], axis=1)
        return data

    def train(self, train_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=train_file))
        # df1 = data[data["Attrition"] == 1]
        # df0 = data[data["Attrition"] == 0]
        # df2 = df0.sample(frac=0.2)
        # data = pandas.concat([df2, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.2,
                                                                                                random_state=36)

        self.model = ensemble.RandomForestClassifier(n_estimators=1000, random_state=36, max_depth=5,
                                                     min_samples_leaf=2, class_weight={1: 0.8, 0: 0.25},max_samples=None)
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("理论准确率：", self.model.score(self.X_train, self.y_train))
        print("实际准确率:", score)

        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)
        # important = self.model.feature_importances_
        # import_series = pandas.Series(important,index=self.X_train.columns)
        # print(import_series.sort_values(ascending=True))
        # import_series.sort_values(ascending=True).plot("barh")
        # plt.show()

    def predict(self, predict_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=predict_file))
        self.X_test = data
        pred = self.model.predict(X=self.X_test)
        dataframe = pandas.DataFrame({'result': pred})  # 将ndarray类型的y_pre_linear正常放到这个位置就行了。
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("output.csv", index=False, sep=',')


def test1():
    t = RandomeTree1()
    t.train(train_file="./员工离职预测训练赛/pfm_train.csv")
    t.predict(predict_file="./员工离职预测训练赛/pfm_test.csv")


if __name__ == '__main__':
    test1()
