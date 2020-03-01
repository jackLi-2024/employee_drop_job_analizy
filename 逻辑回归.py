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
from sklearn.model_selection import cross_val_score


class Test():
    def __init__(self):
        pass

    def split_data(self):
        self.data = pandas.read_table(filepath_or_buffer="./员工离职预测训练赛/pfm_train.csv", sep=",")
        data = self.pre_data(data=self.data)
        # df1 = data[data["Attrition"] == 1]
        # df0 = data[data["Attrition"] == 0]
        # data = pandas.concat([df0, df0, df0, df0, df0, df1, df1, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.1,
                                                                                                random_state=123)

    def pre_data(self, data):
        data.loc[data["BusinessTravel"] == "Travel_Rarely", "BusinessTravel"] = 0
        data.loc[data["BusinessTravel"] == "Travel_Frequently", "BusinessTravel"] = 2
        data.loc[data["BusinessTravel"] == "Non-Travel", "BusinessTravel"] = 1

        data.loc[data["Department"] == "Research & Development", "Department"] = 2
        data.loc[data["Department"] == "Sales", "Department"] = 1
        data.loc[data["Department"] == "Human Resources", "Department"] = 0

        data.loc[data["EducationField"] == "Life Sciences", "EducationField"] = 0
        data.loc[data["EducationField"] == "Medical", "EducationField"] = 1
        data.loc[data["EducationField"] == "Other", "EducationField"] = 2
        data.loc[data["EducationField"] == "Technical Degree", "EducationField"] = 3
        data.loc[data["EducationField"] == "Human Resources", "EducationField"] = 4
        data.loc[data["EducationField"] == "Marketing", "EducationField"] = 5

        data.loc[data["JobRole"] == "Manufacturing Director", "JobRole"] = 0
        data.loc[data["JobRole"] == "Laboratory Technician", "JobRole"] = 1
        data.loc[data["JobRole"] == "Sales Executive", "JobRole"] = 2
        data.loc[data["JobRole"] == "Research Scientist", "JobRole"] = 3
        data.loc[data["JobRole"] == "Healthcare Representative", "JobRole"] = 4
        data.loc[data["JobRole"] == "Human Resources", "JobRole"] = 5
        data.loc[data["JobRole"] == "Sales Representative", "JobRole"] = 6
        data.loc[data["JobRole"] == "Research Director", "JobRole"] = 7
        data.loc[data["JobRole"] == "Manager", "JobRole"] = 8

        data.loc[data["Gender"] == "Female", "Gender"] = 1
        data.loc[data["Gender"] == "Male", "Gender"] = 0

        data.loc[data["MaritalStatus"] == "Divorced", "MaritalStatus"] = 0
        data.loc[data["MaritalStatus"] == "Single", "MaritalStatus"] = 1
        data.loc[data["MaritalStatus"] == "Married", "MaritalStatus"] = 2

        data.loc[data["Over18"] == "Y", "Over18"] = 1
        data.loc[data["Over18"] == "N", "Over18"] = 0

        data.loc[data["OverTime"] == "Yes", "OverTime"] = 1
        data.loc[data["OverTime"] == "No", "OverTime"] = 0
        return data

    def train_(self):
        model = linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=10, max_iter=100)
        a = model.fit(self.X_train, self.y_train)
        # print(model.intercept_)
        # print(model.coef_)
        pred = model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("准确率:", score)
        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)

    def kcv(self):
        kfold = model_selection.KFold(n_splits=100, random_state=7)
        modelCV = linear_model.LogisticRegression()
        scoring = 'accuracy'
        results = model_selection.cross_val_score(modelCV, self.X_train, self.y_train, cv=kfold, scoring=scoring)
        print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


class Test2():
    def split_data(self):
        self.data = pandas.read_table(filepath_or_buffer="./员工离职预测训练赛/pfm_train.csv", sep=",")
        data = self.pre_data(data=self.data)
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.1,
                                                                                                random_state=20)

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

    def train_(self):
        model = linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=10, max_iter=100)
        a = model.fit(self.X_train, self.y_train)
        # print(model.intercept_)
        # print(model.coef_)
        pred = model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("准确率:", score)
        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)


class DropJobModel():
    def read_data(self, filepath_or_buffer="./员工离职预测训练赛/pfm_train.csv", sep=","):
        data = pandas.read_table(filepath_or_buffer=filepath_or_buffer, sep=sep)
        return data

    def pre_data(self, data):
        data.loc[data["BusinessTravel"] == "Travel_Rarely", "BusinessTravel"] = 0
        data.loc[data["BusinessTravel"] == "Travel_Frequently", "BusinessTravel"] = 2
        data.loc[data["BusinessTravel"] == "Non-Travel", "BusinessTravel"] = 1

        data.loc[data["Department"] == "Research & Development", "Department"] = 2
        data.loc[data["Department"] == "Sales", "Department"] = 1
        data.loc[data["Department"] == "Human Resources", "Department"] = 0

        data.loc[data["EducationField"] == "Life Sciences", "EducationField"] = 0
        data.loc[data["EducationField"] == "Medical", "EducationField"] = 1
        data.loc[data["EducationField"] == "Other", "EducationField"] = 2
        data.loc[data["EducationField"] == "Technical Degree", "EducationField"] = 3
        data.loc[data["EducationField"] == "Human Resources", "EducationField"] = 4
        data.loc[data["EducationField"] == "Marketing", "EducationField"] = 5

        data.loc[data["JobRole"] == "Manufacturing Director", "JobRole"] = 0
        data.loc[data["JobRole"] == "Laboratory Technician", "JobRole"] = 1
        data.loc[data["JobRole"] == "Sales Executive", "JobRole"] = 2
        data.loc[data["JobRole"] == "Research Scientist", "JobRole"] = 3
        data.loc[data["JobRole"] == "Healthcare Representative", "JobRole"] = 4
        data.loc[data["JobRole"] == "Human Resources", "JobRole"] = 5
        data.loc[data["JobRole"] == "Sales Representative", "JobRole"] = 6
        data.loc[data["JobRole"] == "Research Director", "JobRole"] = 7
        data.loc[data["JobRole"] == "Manager", "JobRole"] = 8

        data.loc[data["Gender"] == "Female", "Gender"] = 1
        data.loc[data["Gender"] == "Male", "Gender"] = 0

        data.loc[data["MaritalStatus"] == "Divorced", "MaritalStatus"] = 0
        data.loc[data["MaritalStatus"] == "Single", "MaritalStatus"] = 1
        data.loc[data["MaritalStatus"] == "Married", "MaritalStatus"] = 2

        data.loc[data["Over18"] == "Y", "Over18"] = 1
        data.loc[data["Over18"] == "N", "Over18"] = 0

        data.loc[data["OverTime"] == "Yes", "OverTime"] = 1
        data.loc[data["OverTime"] == "No", "OverTime"] = 0
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
                                                                                                random_state=123)

        self.model = linear_model.LogisticRegression(C=0.01, max_iter=100)
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("准确率:", score)
        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)

    def predict(self, predict_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=predict_file))
        self.X_test = data
        pred = self.model.predict(X=self.X_test)
        dataframe = pandas.DataFrame({'result': pred})  # 将ndarray类型的y_pre_linear正常放到这个位置就行了。
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("output.csv", index=False, sep=',')


class DropJobModel1():
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
        # df1 = data[data["Attrition"] == 1]
        # df0 = data[data["Attrition"] == 0]
        # df2 = df0.sample(frac=0.2)
        # data = pandas.concat([df2, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.2,
                                                                                                random_state=36)

        self.model = linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=10, max_iter=100)
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("准确率:", score)
        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)

    def predict(self, predict_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=predict_file))
        self.X_test = data
        pred = self.model.predict(X=self.X_test)
        dataframe = pandas.DataFrame({'result': pred})  # 将ndarray类型的y_pre_linear正常放到这个位置就行了。
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("output.csv", index=False, sep=',')


def test():
    djm = Test()
    djm.split_data()
    djm.train_()


def test1():
    djm = DropJobModel()
    djm.train(train_file="./员工离职预测训练赛/pfm_train.csv")
    djm.predict(predict_file="./员工离职预测训练赛/pfm_test.csv")


def test2():
    djm = Test2()
    djm.split_data()
    djm.train_()


def test3():
    djm = DropJobModel1()
    djm.train(train_file="./员工离职预测训练赛/pfm_train.csv")
    djm.predict(predict_file="./员工离职预测训练赛/pfm_test.csv")


if __name__ == '__main__':
    test3()
