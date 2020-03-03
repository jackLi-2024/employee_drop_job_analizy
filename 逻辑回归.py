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
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC


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
        # df1 = data[data["Attrition"] == 1]
        # df0 = data[data["Attrition"] == 0]
        # df2 = df0.sample(frac=0.2)
        # data = pandas.concat([df2, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.01,
                                                                                                random_state=27)

        self.cv()
        self.model = linear_model.LogisticRegression(solver="liblinear", penalty="l1", C=10, max_iter=100)
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

    def cv(self):
        self.model = linear_model.LogisticRegression(solver="liblinear")
        loss = cross_val_score(self.model, self.X_train, self.y_train, cv=3, scoring='neg_log_loss')
        print('cv accuracy score is:', -loss)
        print('cv logloss is:', -loss.mean())
        penaltys = ['l1', 'l2']
        Cs = [0.001, 0.1, 1, 10, 100, 1000]
        max_iter = [1, 10, 100, 1000, 10000]
        # 调优的参数集合，搜索网格为2x5，在网格上的交叉点进行搜索
        tuned_parameters = dict(penalty=penaltys, C=Cs, max_iter=max_iter)

        grid = GridSearchCV(self.model, tuned_parameters, cv=5, scoring='neg_log_loss', n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        pred = grid.predict(X=self.X_test)
        print("params={} scores={}".format(grid.best_params_, grid.best_score_))

    def predict(self, predict_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=predict_file))
        self.X_test = data
        pred = self.model.predict(X=self.X_test)
        dataframe = pandas.DataFrame({'result': pred})  # 将ndarray类型的y_pre_linear正常放到这个位置就行了。
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("output.csv", index=False, sep=',')


def deal_age(data):
    if 25 <= int(data) < 30:
        return 2
    elif 30 <= int(data) < 40:
        return 3
    elif 40 <= int(data) < 50:
        return 4
    elif 50 <= int(data) < 60:
        return 5
    else:
        return 6


def deal_income(data):
    data = int(data)

    if data < 10000:
        return 4
    elif data < 15000:
        return 5
    elif data < 20000:
        return 6
    elif data < 50000:
        return 7
    else:
        return 8


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
        # data["age"] = data["Age"].apply(deal_age)
        # data["income"] = data["MonthlyIncome"].apply(deal_income)
        # print(data.columns)
        # new_data = data.loc[:, ["Age", "MonthlyIncome", "PercentSalaryHike",
        #                         #                         "PerformanceRating",
        #                         #                         "RelationshipSatisfaction",
        #                         #                         "StockOptionLevel",
        #                         #                         "TotalWorkingYears",
        #                         #                         "TrainingTimesLastYear",
        #                         #                         "WorkLifeBalance",
        #                         #                         "YearsAtCompany",
        #                         #                         "YearsInCurrentRole"
        #                         ]]
        # new_data = (new_data - new_data.mean()) / new_data.std()
        # # new_data = (new_data - new_data.min()) / (new_data.max() - new_data.min())
        # data.drop(["Age", "MonthlyIncome", "EmployeeNumber", "PercentSalaryHike",
        #            # "PerformanceRating",
        #            # "RelationshipSatisfaction",
        #            # "StandardHours",
        #            # "StockOptionLevel",
        #            # "TotalWorkingYears",
        #            # "TrainingTimesLastYear",
        #            # "WorkLifeBalance",
        #            # "YearsAtCompany",
        #            # "YearsInCurrentRole"
        #            ], axis=1, inplace=True)
        # data = pandas.concat([data, new_data], axis=1)
        # data.drop("Y", axis=1, inplace=True)
        # data.drop("StandardHours", axis=1, inplace=True)
        # data.drop("Manager", axis=1, inplace=True)
        # data.drop("PerformanceRating", axis=1, inplace=True)
        # data.drop("Other", axis=1, inplace=True)
        # data.drop("Non-Travel", axis=1, inplace=True)
        # data.drop("Travel_Rarely", axis=1, inplace=True)
        # data.drop("Medical", axis=1, inplace=True)
        # data.drop("Marketing", axis=1, inplace=True)
        # data.drop("Married", axis=1, inplace=True)
        # data.drop("Sales", axis=1, inplace=True)
        # data.drop("Male", axis=1, inplace=True)
        # data.drop("Female", axis=1, inplace=True)
        # data.drop("Travel_Frequently", axis=1, inplace=True)
        # data.drop("Divorced", axis=1, inplace=True)
        # data.drop("TrainingTimesLastYear", axis=1, inplace=True)
        # data.drop("EnvironmentSatisfaction", axis=1, inplace=True)
        # data.drop("WorkLifeBalance", axis=1, inplace=True)
        # data.drop("JobInvolvement", axis=1, inplace=True)
        # data.drop("Education", axis=1, inplace=True)
        # data.drop("RelationshipSatisfaction", axis=1, inplace=True)
        # data.drop("YearsSinceLastPromotion", axis=1, inplace=True)
        # data = data.loc[:,
        #        ["Attrition", "Age", "MonthlyIncome", "PercentSalaryHike", "YearsAtCompany", "TotalWorkingYears"]]

        return data

    def train(self, train_file):
        data = self.pre_data(data=self.read_data(filepath_or_buffer=train_file))
        # df1 = data[data["Attrition"] == 1]
        # df0 = data[data["Attrition"] == 0]
        # df2 = df0.sample(frac=0.2)
        # data = pandas.concat([df2, df1])
        y = data.Attrition
        X = data.drop(["Attrition"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.4,
                                                                                                random_state=149)

        # self.cv()
        self.model = linear_model.LogisticRegression(solver="liblinear", penalty="l2", C=1, max_iter=100,
                                                     random_state=149, class_weight={1:0.55,0:0.5}, tol=0.00001)
        self.model.fit(self.X_train, self.y_train, sample_weight=100)
        pred = self.model.predict(X=self.X_test)
        # print(pandas.Series(pred).value_counts())
        score = metrics.accuracy_score(self.y_test, pred)
        score_true = metrics.recall_score(self.y_test, pred)
        score_false = metrics.recall_score(self.y_test, pred, pos_label=0)
        print("理论准确率：", self.model.score(self.X_train, self.y_train))
        print("实际准确率:", score)
        print("正例覆盖率：", score_true)
        print("负例覆盖率：", score_false)

    def cv(self):
        self.model = linear_model.LogisticRegression(solver="lbfgs")

        penaltys = ['l1', 'l2']
        Cs = [0.001, 0.1, 1, 10, 100, 1000]
        max_iter = [10, 100, 1000, 10000]
        # 调优的参数集合，搜索网格为2x5，在网格上的交叉点进行搜索
        tuned_parameters = dict(penalty=penaltys, C=Cs, max_iter=max_iter)

        grid = GridSearchCV(self.model, tuned_parameters, cv=5, scoring='neg_log_loss', n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        loss = cross_val_score(grid, self.X_train, self.y_train, cv=5, scoring='neg_log_loss')
        print('cv accuracy score is:', -loss)
        print('cv logloss is:', -loss.mean())
        pred = grid.predict(X=self.X_test)
        print("params={} scores={} n={}".format(grid.best_params_, grid.best_score_, grid.best_estimator_))

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
    # djm.predict(predict_file="./员工离职预测训练赛/pfm_test.csv")


if __name__ == '__main__':
    # test1()
    test3()
