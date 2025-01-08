from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score, accuracy_score)
from numpy import sqrt, mean, std, array, arange
from mlworkflow.sanitation import check_strs, is_real_valued
from mlworkflow.utils import print_cond
import matplotlib.pyplot as plt


class Metrics:
    """
    This function encapsulates several metrics and calculates them for a vector of true values versus a vector of
    predictions
    """
    def __init__(self, default_cont='rmse', default_cat='acc', plot_res=False):
        """
        The class has two attributes: cont_ops for the metrics for continuous variables and cat_ops for categorical
        variables. New metrics are added as lambdas to both these dictionaries
        """
        self.cont_ops = {
            'mse': (lambda orig, pred: mean_squared_error(orig, pred)),
            'rmse': (lambda orig, pred: sqrt(mean_squared_error(orig, pred))),
            'mae': (lambda orig, pred: mean_absolute_error(orig, pred)),
            'mape': (lambda orig, pred: mean_absolute_percentage_error(orig, pred)),
            'r2': (lambda orig, pred: r2_score(orig, pred))
        }

        self.cat_ops = {
            'acc': (lambda orig, pred: accuracy_score(orig, pred)),
        }

        self.default_cont = default_cont
        self.default_cat = default_cat
        self.plot_res = plot_res

    @staticmethod
    def __check_metric(key, ops):
        if key not in ops:
            raise KeyError('Unknown metric ' + key)

    @staticmethod
    def __check_operators(key, ops):
        if key not in ops:
            raise KeyError('Unknown metric ' + key)

    def calc(self, orig, pred, op=None, print_res=True):
        """
        Calculate the metric named in op for the values in orig and pred
        :param orig: the list or vector with the real values
        :param pred: the list or vector with the predicted values
        :param op: string with the name of the operator to use
        :param print_res: boolean that defines whether to print the results
        :return: the calculated value of the metric
        """
        if op is not None:
            check_strs(op)

        if is_real_valued(orig):
            if op is None:
                op = self.default_cont
            else:
                self.__check_metric(op.lower(), self.cont_ops)
            res = self.cont_ops[op.lower()](orig, pred)
        else:
            if op is None:
                op = self.default_cat
            else:
                self.__check_metric(op.lower(), self.cat_ops)
            res = self.cat_ops[op.lower()](orig, pred)

        print_cond(print_res, op.lower() + ': ' + str(res))
        self.plot_cond(orig, pred)

        return res

    @staticmethod
    def __calc_set(orig, pred, op_set, print_res):
        """
        Calculate all metrics inside op_set for the values in orig and pred
        :param orig: the list or vector with the real values
        :param pred: the list or vector with the predicted values
        :param op_set: dictionary with metrics in the form name: lambda
        :print_res: boolean that defines whether to print the results
        :return: the calculated metrics inside a dictionary
        """
        res = {}
        for op in op_set:
            res.update({op: op_set[op](orig, pred)})
            print_cond(print_res, op + ': ' + str(res[op]))

        return res

    def calc_all(self, orig, pred, print_res=True):
        """
        Calculate all known metrics for the values in orig and pred
        :param orig: the list or vector with the real values
        :param pred: the list or vector with the predicted values
        :param print_res: boolean that defines whether to print the results
        :return: the calculated metrics inside a dictionary
        """
        if is_real_valued(orig):
            res = self.__calc_set(orig, pred, self.cont_ops, print_res)
        else:
            res = self.__calc_set(orig, pred, self.cat_ops, print_res)

        return res

    @staticmethod
    def plot_preds(orig, pred, y_label='', x_names=[]):
        plt.figure()
        plt.plot(orig, marker='o', color='black', linestyle='None')
        plt.plot(pred, marker='x', color='red', linestyle='None')
        plt.ylabel(y_label)
        if len(x_names) > 0:
            for (xi, yi) in zip(list(range(len(orig))), orig):
                plt.text(xi, yi, x_names[xi], va='bottom', ha='center')
        plt.show()

    @staticmethod
    def plot_residuals(orig, pred, y_label='', x_names=[]):
        residuals = orig - pred
        resid_mean = array([mean(residuals)] * len(residuals))
        resid_std = array([std(residuals)] * len(residuals))
        plt.figure()
        plt.plot(arange(len(residuals)), residuals, marker='o', color='black', linestyle='None')
        plt.plot(resid_mean, marker='None', color='red', linestyle='solid')
        plt.plot(resid_mean - 2 * resid_std, marker='None', color='purple', linestyle='solid')
        plt.plot(resid_mean + 2 * resid_std, marker='None', color='purple', linestyle='solid')
        plt.ylabel(y_label)
        if len(x_names) > 0:
            for (xi, yi) in zip(list(range(len(residuals))), residuals):
                plt.text(xi, yi, x_names[xi], va='bottom', ha='center')
        plt.show()

    def plot_cond(self, orig, pred):
        if self.plot_res:
            self.plot_preds(orig, pred)


def check_metrics_or_none(obj):
    """
    Sanitation check for a Metrics or None object. Cannot be inside sanitation.py due to circular imports
    """
    if (not isinstance(obj, Metrics)) and (obj is not None):
        raise TypeError(f"either None or Metrics object expected, got {type(obj).__name__}")


def check_metrics(obj):
    """
    Sanitation check for a Metrics object. Cannot be inside sanitation.py due to circular imports
    """
    if not isinstance(obj, Metrics):
        raise TypeError(f"Metrics object expected, got {type(obj).__name__}")
