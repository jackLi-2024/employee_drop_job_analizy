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


class Test():
    def __init__(self):
        pass

    def split_data(self):
        self.data = pandas.read_table(filepath_or_buffer="./员工离职预测训练赛/pfm_train.csv", sep=",")
        data = self.pre_data(data=self.data)
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.2)

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
        model = linear_model.LogisticRegression()
        model.fit(self.X_train, self.y_train)
        # print(model.intercept_)
        # print(model.coef_)
        pred = model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        print("准确率:", score)


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
        self.y_train = data.Attrition
        self.X_train = data.drop(["Attrition"], axis=1)

        self.model = linear_model.LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

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


if __name__ == '__main__':
    test()
