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
import pandas  as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV


class Logistic():
    def read_data(self, filepath_or_buffer="./员工离职预测训练赛/pfm_train.csv", sep=","):
        data = pd.read_table(filepath_or_buffer=filepath_or_buffer, sep=sep)
        return data

    def pre_data(self, data):
        drop_l = []
        for col in data.columns:
            length = len(data[col].unique())
            if length == 1:
                drop_l.append(col)
        data = self.build_feature(data)
        data.drop(drop_l, axis=1, inplace=True)
        data.drop(["EmployeeNumber", "JobLevel"], axis=1, inplace=True)
        data.drop(["Education", "MonthlyIncome", "JobInvolvement", "TotalWorkingYears", "JobSatisfaction",
                   "EnvironmentSatisfaction", "RelationshipSatisfaction"], axis=1, inplace=True)
        data = pd.get_dummies(data)
        return data

    def build_feature(self, data):
        data["AgeEducation"] = data["Education"] * 100 / data["Age"]
        data["AgeIncome"] = data["MonthlyIncome"] * data["JobInvolvement"] * data["TotalWorkingYears"] // data["Age"]
        data["Satisfaction"] = data['JobSatisfaction'] + data['EnvironmentSatisfaction'] + data[
            'RelationshipSatisfaction']
        return data

    def CV(self):
        params = {
            'penalty': ['l1', 'l2'],
            'C': np.arange(1, 4.1, 0.2),
        }
        estimator = linear_model.LogisticRegression(solver='liblinear')
        grid = GridSearchCV(estimator, param_grid=params, cv=10)
        grid.fit(self.X_train, self.y_train)
        print('最好的参数', grid.best_params_)
        score = grid.score(self.X_test, self.y_test)
        print('在测试集的得分', score)

    def train(self, train_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=train_file))
        # df1 = data[data["Attrition"] == 1]
        # df0 = data[data["Attrition"] == 0]
        # df2 = df0.sample(frac=0.2)
        # data = pandas.concat([df2, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y,
                                                                                                random_state=27)

        # self.CV()
        self.model = linear_model.LogisticRegression(solver="liblinear", penalty="l2", C=2.4, max_iter=100)
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

    def predict(self, predict_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=predict_file))
        self.X_test = data
        pred = self.model.predict(X=self.X_test)
        dataframe = pd.DataFrame({'result': pred})
        dataframe.to_csv("output.csv", index=False, sep=',')

def test():
    logis = Logistic()
    logis.train(train_file="./员工离职预测训练赛/pfm_train.csv")
    logis.predict(predict_file="./员工离职预测训练赛/pfm_test.csv")

if __name__ == '__main__':
    test()