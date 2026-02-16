from mlworkflow.models.model import Model
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from mlworkflow.utils import print_cond
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt


import numpy as np
import re


class LinRegModel(Model):
    """
    Simple class for a lineal regression model that uses pandas dataframes as input
    """
    def __init__(self, model=LinearRegression, components=[MinMaxScaler()], restrict=False):
        Model.__init__(self, model, components, restrict)

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
        res = res.flatten()

        if self.restrict:
            res[res > 100.0] = 100.0
            res[res < 0.0] = 0.0

        return res

    def feature_imp(self):
        """
        Importance of the features. For the case of lineal regression, the coefficients are returned.
        :return: the coefficients of the model
        """
        return self.pipe.steps[1][1].coef_

    def assign_params(self, x):
        pass


class RidgeModel(LinRegModel):
    """
    Class for a lineal regression model with l2 regularization that uses pandas dataframes as input
    """

    def __init__(self, model=Ridge, components=[MinMaxScaler()]):
        Model.__init__(self, model, components)


class LassoModel(LinRegModel):
    """
    Class for a lineal regression model with l2 regularization that uses pandas dataframes as input
    """

    def __init__(self, model=Lasso, components=[MinMaxScaler()]):
        Model.__init__(self, model, components)


class ElasticModel(LinRegModel):
    """
    Class for a lineal regression model with l1 and l2 regularization that uses pandas dataframes as input
    """

    def __init__(self, model=ElasticNet, alpha=1.0, l1_ratio=0.5, components=[]):
        Model.__init__(self, model, components)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

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
        self.pipe = make_pipeline(*self.components, self.model(alpha=self.alpha, l1_ratio=self.l1_ratio))
        x_train, x_test, y_train, y_test = self.split_data(x, y, split)
        self.pipe.fit(x_train, y_train)
        print_cond(print_res, "Fit score: " + str(self.pipe.score(x_test, y_test)))

        return None

    def assign_params(self, x):
        """
        Assign the parameters of the SVR model with a list of values in the form [degree, gamma, coef0, C, epsilon].
        The type of kernel is defined when creating the model, not with this function. This function is needed in
        order to apply parameter optimization
        :param x: a list of values to be assigned as parameters
        """
        self.alpha = x[0]
        self.l1_ratio = x[1]


class WeightedLinRegModel(LinRegModel):
    """
    Class for a lineal regression model with weights that uses pandas dataframes as input
    """

    def __init__(self, model=LinearRegression, components=[MinMaxScaler()], weights=None):
        Model.__init__(self, model, components)
        self.weights = weights

    def fit(self, df, obj_var, feature_var, split=False, print_res=True):
        """
        Fit the linear regression model to the data inside df for the objective variable obj_var using the feature
        variables feature_var and weighting the training instances
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param feature_var: a list with the names of the feature variables
        :param split: a boolean that determines whether to do a train/test split
        :param print_res: boolean that defines whether to print the results
        """
        self.obj_var = obj_var
        self.feature_var = feature_var
        x, y = self.from_pd_to_np(df)
        if self.weights is None:
            self.weights = [20.0] * x.shape[0]  # If no weights are provided, uniform weights
        self.pipe = make_pipeline(*self.components, self.model())
        x_train, x_test, y_train, y_test = self.split_data(x, y, split)
        self.pipe.fit(x_train, y_train, linearregression__sample_weight=self.weights)
        print_cond(print_res, "Fit score: " + str(self.pipe.score(x_test, y_test)))

        return None


class RFModel(Model):
    """
    Class for a random forest regression model that uses pandas dataframes as input
    """
    def __init__(self, model=RandomForestRegressor, components=[MinMaxScaler()], restrict=False, n_estimators=15,
                 max_depth=4, min_samples_split=5, min_samples_leaf=3):
        Model.__init__(self, model, components, restrict)
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

        if self.restrict:
            res[res > 100.0] = 100.0
            res[res < 0.0] = 0.0

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


class SVRModel(Model):
    """
    Class for a support vector regression model that uses pandas dataframes as input
    """

    def __init__(self, model=SVR, components=[MinMaxScaler()], restrict=False, kernel='poly', degree=3,
                 gamma='scale', coef0=0.0, C=1.0, epsilon=0.1):
        Model.__init__(self, model, components, restrict)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        self.epsilon = epsilon

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
                                             coef0=self.coef0, C=self.C, epsilon=self.epsilon))
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

        if self.restrict:
            res[res > 100.0] = 100.0
            res[res < 0.0] = 0.0

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
        self.epsilon = x[4]


class GaussianProcessModel(Model):
    """
    Class for a Gaussian process regression model that uses pandas dataframes as input
    """

    def __init__(self, model=GaussianProcessRegressor, components=[], restrict=False, kernel=RBF(), length_scale=1.0,
                 length_scale_bounds=(1e-25, 1e10), n_restarts_optimizer=10):
        Model.__init__(self, model, components, restrict)
        self.kernel = kernel
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.n_restarts_optimizer = n_restarts_optimizer

    @staticmethod
    def fix_kernel_size(e):
        """ Depending on components like PolynomialFeatures or PCA, the size of the kernel changes """
        size = re.search('\d+!=\d+', e)[0].split('!=')[1]

        return int(size)

    def fit(self, df, obj_var, feature_var, split=False, print_res=True):
        """
        Fit the Gaussian process model to the data inside df for the objective variable obj_var using the
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
        x_train, x_test, y_train, y_test = self.split_data(x, y, split)
        size = x.shape[1]

        try:
            self.fit_kernel(size, x_train, y_train)
        except ValueError as e:
            size = self.fix_kernel_size(e.__str__())  # In case components change the size of the data in the pipeline
            self.fit_kernel(size, x_train, y_train)

        print_cond(print_res, "Fit score: " + str(self.pipe.score(x_test, y_test)))

        return None

    def fit_kernel(self, size, x_train, y_train):
        kernel = RBF(self.length_scale * np.ones(size), self.length_scale_bounds)
        kernel = ConstantKernel(self.length_scale, self.length_scale_bounds) * \
                 RBF(self.length_scale * np.ones(size), self.length_scale_bounds) + \
                 WhiteKernel(self.length_scale, self.length_scale_bounds)
        kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-3, 1e4)) + \
                       WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e1))
        self.pipe = make_pipeline(*self.components,
                                  self.model(kernel=kernel, n_restarts_optimizer=self.n_restarts_optimizer,
                                             alpha=1.0, random_state=42))
        self.pipe.fit(x_train, y_train)

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

        if self.restrict:
            res[res > 100.0] = 100.0
            res[res < 0.0] = 0.0

        return res

    def predict_confidence(self, df):
        """
        Predict the values of the instances inside the dataframe df. Can apply all the metrics inside a Metrics object
        if provided. It can also take a dataframe with no objective variable column, in case we are applying the model
        to a real problem outside of testing and we want to predict this value. In this case, this function
        shows a plot with the 95% confidence intervals of each prediction.
        :param df: the pandas dataframe with the data
        :return: the predicted values inside a numpy array
        """
        x, _ = self.from_pd_to_np(df, strict=False)

        res, std_prediction = self.pipe.predict(x, return_std=True)

        low_range = res - 1.96 * std_prediction
        high_range = res + 1.96 * std_prediction
        plt.plot(range(len(res)), res, linestyle='None', marker='o', color='blue')
        plt.fill_between(range(len(res)),
                         low_range,
                         high_range,
                         alpha=0.5,
                         label=r"95% confidence interval",
                         )
        plt.plot(range(len(res)), [0] * len(res), color='red')

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
        Assign the parameters of the Gaussian process model with a list of values in the form [length_scale,
        length_scale_bounds_0, length_scale_bounds_1].
        The type of kernel is defined when creating the model, not with this function. This function is needed in
        order to apply parameter optimization
        :param x: a list of values to be assigned as parameters
        """
        self.length_scale = x[0]
        self.length_scale_bounds = (x[1], x[2])
