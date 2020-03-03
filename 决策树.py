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


class Tree1():
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
        return data

    def train(self, train_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=train_file))
        df1 = data[data["Attrition"] == 1]
        df0 = data[data["Attrition"] == 0]
        df2 = df0.sample(frac=0.2)
        data = pandas.concat([df2, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.2,
                                                                                                random_state=36)
        max_depth = [2, 3, 4, 5, 6]
        min_samples_split = [2, 4, 6, 8]
        min_samples_leaf = [2, 4, 8, 10, 12,14,16]
        parameters = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        grid = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=parameters, cv=10)
        grid.fit(self.X_train, self.y_train)
        print(grid.best_params_)
        # {'max_depth': 3, 'min_samples_leaf': 14, 'min_samples_split': 2}
        self.model = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, min_impurity_split=2)
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
        dataframe = pandas.DataFrame({'result': pred})  # 将ndarray类型的y_pre_linear正常放到这个位置就行了。
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("output.csv", index=False, sep=',')


def test1():
    t = Tree1()
    t.train(train_file="./员工离职预测训练赛/pfm_train.csv")


if __name__ == '__main__':
    test1()
