import numpy as np
from abc import ABC, abstractmethod
from mlworkflow.sanitation import check_pd, check_vars
from mlworkflow.models.metrics import check_metrics_or_none
from mlworkflow.utils import print_cond
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from numpy import mean
import pickle
from datetime import datetime


class Model(ABC):
    """
    Abstract class that defines a machine learning model inherited by specific models.
    """
    def __init__(self, model, components=[], restrict=False):
        self.pipe = None
        self.obj_var = None
        self.feature_var = None
        self.model = model
        self.components = components
        self.restrict = restrict

    @abstractmethod
    def fit(self, df, obj_var, feature_var, split, print_res):
        pass

    @abstractmethod
    def predict(self, df):
        pass

    @abstractmethod
    def feature_imp(self):
        pass

    def to_str(self):
        return 'Model characteristics:\n' + str(self.__dict__)

    def from_pd_to_np(self, df, strict=True):
        """
        Transform a pandas dataframe into a tuple of numpy vectors X y with features and objective variables
        for sklearn pipelines
        :param df: the original pandas dataframe
        :param strict: a boolean that decides whether to strictly check for the objective variable inside df
        :return: a pair X, y of numpy vectors
        """
        check_pd(df)
        check_vars(df, self.feature_var)
        if strict:
            check_vars(df, self.obj_var)

        x = np.array(df[self.feature_var]).reshape(-1, len(self.feature_var))
        if not all(list(map(lambda v: v in df, self.obj_var))):
            y = None
        else:
            y = np.array(df[self.obj_var])

        return x, y

    @staticmethod
    def split_data(x, y, split=True):
        if split:
            x_train, x_test, y_train, y_test = train_test_split(x, y)
        else:
            x_train, x_test, y_train, y_test = x, x, y, y

        return x_train, x_test, y_train, y_test

    def eval_data(self, df, train_index, test_index, metrics, print_res):
        """
        Fit the pipeline to the x data and predict the y values to evaluate
        """
        self.fit(df.iloc[train_index], self.obj_var, self.feature_var, split=False, print_res=print_res)
        preds = self.predict(df.iloc[test_index])
        _, y = self.from_pd_to_np(df.iloc[test_index], strict=False)
        score = metrics.calc(y, preds, print_res=print_res)

        return score, preds

    def eval_cv(self, df, obj_var, feature_var, metrics, folds=5,
                shuffle=False, seed=None, splits=None, print_res=True, ret_preds=False):
        """
        Run a cross-validation with the model over some data and return the score
        """
        check_metrics_or_none(metrics)

        self.obj_var = obj_var
        self.feature_var = feature_var
        res_score = []
        orig = []
        res_preds = []

        if splits is None:
            kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
            splits = kf.split(df)

        for i, (train_index, test_index) in enumerate(splits):
            print_cond(print_res, f"Fold {i}")
            score, preds = self.eval_data(df, train_index, test_index, metrics, print_res=print_res)
            res_score.append(score)
            orig = orig + list(df.iloc[test_index][obj_var].values.flatten())
            res_preds = res_preds + list(preds)

        res = mean(res_score)
        if ret_preds:
            res = (res, (orig, res_preds))

        return res

    def eval_loo(self, df, obj_var, feature_var, metrics, compounds, params=[], print_res=True, plot_res=True):
        """
        Run a leave-one-out cross-validation with the model over some data and return the score. The names of the
        compounds are expected to be in the same order as the dataframe
        """
        check_metrics_or_none(metrics)

        self.obj_var = obj_var
        self.feature_var = feature_var
        res_score = []
        y = df[obj_var[0]].tolist()
        res_preds = []
        if len(params) > 0:
            self.assign_params(params)
        length = range(len(compounds))

        for i in length:
            print_cond(print_res, f"Compound {compounds[i]}")
            train_index = list(length)
            del train_index[i]
            test_index = [i]
            score, preds = self.eval_data(df, train_index, test_index, metrics, print_res=print_res)
            res_score.append(score)
            res_preds.append(preds[0])

        print_cond(print_res, f'Average error obtained: {np.mean(res_score)}')
        print_cond(print_res, f'Standard deviation of the error: {np.std(res_score)}')
        if plot_res:
            metrics.plot_preds(y, res_preds, y_label=obj_var[0], x_names=compounds)

        residuals = df[obj_var[0]] - res_preds
        print_cond(print_res, f'Mean of the residuals: {np.mean(residuals)}')
        print_cond(print_res, f'Standard deviation of the residuals: {np.std(residuals)}')
        if plot_res:
            metrics.plot_residuals(df[obj_var[0]], res_preds, obj_var[0], compounds)

        return res_score, res_preds

    @abstractmethod
    def assign_params(self, x):
        """
        Abstract method for assigning a list of values x as the parameters of the model. Important function for
        hyperparameter optimization
        :param x: a list of values to be assigned as parameters
        """
        pass

    def save_model(self):
        date = datetime.now()
        filename = str(self.__class__.__name__) + date.strftime('_%d_%m_%Y_%H_%M.pkl')
        with open(filename, 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)


def check_model(obj):
    """
    Sanitation check for a Model object. Cannot be inside sanitation.py due to circular imports
    """
    if not issubclass(obj, Model):
        raise TypeError(f"Model object expected, got {type(obj).__name__}")


def check_model_instance(obj):
    """
    Sanitation check for a Model instance. Cannot be inside sanitation.py due to circular imports
    """
    if not isinstance(obj, Model):
        raise TypeError(f"Model instance expected, got {type(obj).__name__}")