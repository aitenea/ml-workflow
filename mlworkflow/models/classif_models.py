from mlworkflow.models.model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from mlworkflow.utils import print_cond
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt

import numpy as np
import re


class LogRegModel(Model):
    """
    Simple class for a lineal regression model that uses pandas dataframes as input
    """
    def __init__(self, model=LogisticRegression, components=[MinMaxScaler()]):
        Model.__init__(self, model, components)

    def fit(self, df, obj_var, feature_var, split=False, print_res=True):
        """
        Fit the linear regression model to the data inside df for the objective variable obj_var using the feature
        variables feature_var
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param feature_var: a list with the names of the feature variables
        :param split: a boolean that determines whether to do a train/test split
        :param print_res: boolean that defines whether to print the results
        """
        self.obj_var = obj_var
        self.feature_var = feature_var
        x, y = self.from_pd_to_np(df)
        self.pipe = make_pipeline(*self.components, self.model())
        x_train, x_test, y_train, y_test = self.split_data(x, y, split)
        self.pipe.fit(x_train, y_train.ravel())
        print_cond(print_res, "Fit score: " + str(self.pipe.score(x_test, y_test)))

        return None

    def predict(self, df):
        """
        Predict the values of the instances inside the dataframe df. Can apply all the metrics inside a Metrics object
        if provided. It can also take a dataframe with no objective variable column, in case we are applying the model
        to a real problem outside of testing and we want to predict this value.
        :param df: the pandas dataframe with the data
        :return: the predicted values inside a numpy array
        """

        x, _ = self.from_pd_to_np(df, strict=False)
        res = self.pipe.predict(x)
        res = res.flatten()

        return res

    def feature_imp(self):
        """
        Importance of the features. For the case of lineal regression, the coefficients are returned.
        :return: the coefficients of the model
        """
        return self.pipe.steps[1][1].coef_

    def assign_params(self, x):
        pass


class RFCModel(Model):
    """
    Class for a random forest classification model that uses pandas dataframes as input
    """
    def __init__(self, model=RandomForestClassifier, components=[MinMaxScaler()], n_estimators=15,
                 max_depth=4, min_samples_split=5, min_samples_leaf=3):
        Model.__init__(self, model, components)
        # Default values for low data scenario
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, df, obj_var, feature_var, split=False, print_res=True):
        """
        Fit the linear regression model to the data inside df for the objective variable obj_var using the feature
        variables feature_var
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param feature_var: a list with the names of the feature variables
        :param split: a boolean that determines whether to do a train/test split
        :param print_res: boolean that defines whether to print the results
        """
        self.obj_var = obj_var
        self.feature_var = feature_var
        x, y = self.from_pd_to_np(df)
        self.pipe = make_pipeline(*self.components,
                                  self.model(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf))
        x_train, x_test, y_train, y_test = self.split_data(x, y, split)
        y_train = y_train.reshape(1, -1)[0]  # The random forest needs a 1d vector instead of a column vector
        y_test = y_test.reshape(1, -1)[0]
        self.pipe.fit(x_train, y_train)
        print_cond(print_res, self.pipe.score(x_test, y_test))

        return None

    def predict(self, df):
        """
        Predict the values of the instances inside the dataframe df. Can apply all the metrics inside a Metrics object
        if provided. It can also take a dataframe with no objective variable column, in case we are applying the model
        to a real problem outside of testing and we want to predict this value.
        :param df: the pandas dataframe with the data
        :return: the predicted values inside a numpy array
        """

        x, _ = self.from_pd_to_np(df, strict=False)
        res = self.pipe.predict(x)

        return res

    def feature_imp(self):
        """
        Return importance of the features inside the model
        :return: the coefficients of the model
        """
        pass
        # return self.pipe.steps[1][1].coef_

    def assign_params(self, x):
        """
        Assign the parameters of the SVR model with a list of values in the form
        [n_estimators, max_depth, min_samples_split, min_samples_leaf].
        This function is needed in order to apply parameter optimization
        :param x: a list of values to be assigned as parameters
        """
        self.n_estimators = int(x[0])
        self.max_depth = int(x[1])
        self.min_samples_split = int(x[2])
        self.min_samples_leaf = int(x[3])


class SVCModel(Model):
    """
    Class for a support vector classification model that uses pandas dataframes as input
    """

    def __init__(self, model=SVC, components=[MinMaxScaler()], kernel='poly', degree=3,
                 gamma='scale', coef0=0.0, C=1.0):
        Model.__init__(self, model, components)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C

    def fit(self, df, obj_var, feature_var, split=False, print_res=True):
        """
        Fit the support vector regression model to the data inside df for the objective variable obj_var using the
        feature variables feature_var
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param feature_var: a list with the names of the feature variables
        :param split: a boolean that determines whether to do a train/test split
        :param print_res: boolean that defines whether to print the results
        """
        self.obj_var = obj_var
        self.feature_var = feature_var
        x, y = self.from_pd_to_np(df)
        self.pipe = make_pipeline(*self.components,
                                  self.model(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                                             coef0=self.coef0, C=self.C))
        x_train, x_test, y_train, y_test = self.split_data(x, y, split)
        y_train = y_train.reshape(1, -1)[0]  # The SVR needs a 1d vector instead of a column vector
        y_test = y_test.reshape(1, -1)[0]
        self.pipe.fit(x_train, y_train)
        print_cond(print_res, "Fit score: " + str(self.pipe.score(x_test, y_test)))

        return None

    def predict(self, df):
        """
        Predict the values of the instances inside the dataframe df. Can apply all the metrics inside a Metrics object
        if provided. It can also take a dataframe with no objective variable column, in case we are applying the model
        to a real problem outside of testing and we want to predict this value.
        :param df: the pandas dataframe with the data
        :return: the predicted values inside a numpy array
        """

        x, _ = self.from_pd_to_np(df, strict=False)
        res = self.pipe.predict(x)

        return res

    def feature_imp(self):
        """
        Return importance of the features inside the model
        :return: the coefficients of the model
        """
        pass
        # return self.pipe.steps[1][1].coef_

    def assign_params(self, x):
        """
        Assign the parameters of the SVR model with a list of values in the form [degree, gamma, coef0, C, epsilon].
        The type of kernel is defined when creating the model, not with this function. This function is needed in
        order to apply parameter optimization
        :param x: a list of values to be assigned as parameters
        """
        self.degree = round(x[0])
        self.gamma = x[1]
        self.coef0 = x[2]
        self.C = x[3]