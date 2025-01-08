from mlworkflow.utils import print_cond
from mlworkflow.models.model import check_model
from mlworkflow.models.metrics import check_metrics
from mlworkflow.sanitation import check_pd, check_strs, check_vars


class GreedyVarSel:
    """
    Simple greedy forward algorithm for variable selection. The algorithm uses cross-validation to test the different
    feature sets, so it can arrive to different results from the same starting point. By default, the cross-validation
    is deterministic without shuffle, and so is the algorithm
    """
    def __init__(self, model, metrics, components=[], max_features=20, force=False):
        check_model(model)
        check_metrics(metrics)

        self.model = model
        self.metrics = metrics
        self.vars = []
        self.best_err = float('inf')
        self.max_features = max_features
        self.force = force
        self.components = components
        self.folds = None
        self.shuffle = None
        self.seed = None

    def run(self, df, obj_var, ini_feat=[], folds=5, shuffle=False, seed=None, print_res=True):
        """
        Run the search algorithm to find the best greedy forward variable set and its error
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param ini_feat: a list with the names of the initial set of variables. Default to empty list
        :param folds: number of folds for the cross-validation
        :param shuffle: whether to shuffle the instances during the folding or not.
        This transforms the algorithm into a non-deterministic one if set to True.
        :param seed: the seed for the random state if shuffle is set to True
        :param print_res: boolean that defines whether to print the results
        :return: the obtained best feature set and its error
        """
        check_pd(df)
        check_strs(*obj_var)
        check_vars(df, obj_var)
        self.folds = folds
        self.shuffle = shuffle
        self.seed = seed
        if len(ini_feat) != 0:
            check_strs(*ini_feat)
            check_vars(df, ini_feat)
            self.vars = ini_feat
            self.best_err = self.eval_feature(df, obj_var, None)

        end = False

        while not end and len(self.vars) < self.max_features:
            print_cond(print_res, f"Current feature set: {self.vars}")
            print_cond(print_res, f"Current best error: {self.best_err}")
            feature, err = self.search_step(df, obj_var)
            print_cond(print_res, f"Best variable found: {feature}, with error {err}")
            if err < self.best_err or self.force:
                self.best_err = err
                self.vars.append(feature)
            else:
                end = True

        print_cond(print_res, f"Best feature set found: {self.vars}")

        return self.vars, self.best_err

    def to_str(self):
        return 'Greedy algorithm characteristics:\n' + str(self.__dict__)

    def search_step(self, df, obj_var):
        """
        Execute a single search step in the algorithm: evaluate adding a single feature from all possible ones to
        the feature set and choose the one with better error after testing all of them
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :return: the best feature set found and its error
        """
        idx = df.columns.difference(self.vars + obj_var)
        best_feature = None
        best_err = float('inf')
        for i in idx:
            err = self.eval_feature(df, obj_var, i)
            if err < best_err:
                best_err = err
                best_feature = i

        return best_feature, best_err

    def eval_feature(self, df, obj_var, col):
        """
        Evaluate the performance of adding a feature to the feature set
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param col: the feature to evaluate
        :return: the average error obtained from evaluating the cross-validation
        """
        m = self.model(components=self.components)
        if col is None:
            feature_var = self.vars
        else:
            feature_var = self.vars + [col]
        err = m.eval_cv(df, obj_var, feature_var, self.metrics,
                        self.folds, self.shuffle, self.seed, print_res=False)

        return err


class BackwardsGreedyVarSel(GreedyVarSel):
    """
    Simple greedy backwards algorithm for variable selection. The algorithm uses cross-validation to test the different
    feature sets, so it can arrive to different results from the same starting point. By default, the cross-validation
    is deterministic without shuffle, and so is the algorithm
    """
    def __init__(self, model, metrics, components=[], force=True, min_features=20):
        GreedyVarSel.__init__(self, model=model, metrics=metrics, components=components, force=force)
        self.min_features = min_features

    def run(self, df, obj_var, ini_feat=[], folds=5, shuffle=False, seed=None, print_res=True):
        """
        Run the search algorithm to find the best greedy backwards variable set and its error
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param ini_feat: a list with the names of the initial set of variables. Default to empty list
        :param folds: number of folds for the cross-validation
        :param shuffle: whether to shuffle the instances during the folding or not.
        This transforms the algorithm into a non-deterministic one if set to True.
        :param seed: the seed for the random state if shuffle is set to True
        :param print_res: boolean that defines whether to print the results
        :return: the obtained best feature set and its error
        """
        check_pd(df)
        check_strs(*obj_var)
        check_vars(df, obj_var)
        self.folds = folds
        self.shuffle = shuffle
        self.seed = seed
        if len(ini_feat) != 0:
            check_strs(*ini_feat)
            check_vars(df, ini_feat)
            self.vars = ini_feat
            self.best_err = self.eval_feature(df, obj_var, None)
        else:
            self.vars = df.columns.difference(obj_var)

        end = False

        while not end and len(self.vars) > self.min_features:
            print_cond(print_res, f"Current feature set: {self.vars}")
            print_cond(print_res, f"Current best error: {self.best_err}")
            feature, err = self.search_step(df, obj_var)
            print_cond(print_res, f"Worst variable found: {feature}, with error {err}")
            if err < self.best_err or self.force:
                self.best_err = err
                self.vars = self.vars.difference([feature])
            else:
                end = True

        print_cond(print_res, f"Best feature set found: {self.vars}")

        return self.vars, self.best_err

    def to_str(self):
        return 'Backwards greedy algorithm characteristics:\n' + str(self.__dict__)

    def search_step(self, df, obj_var):
        """
        Execute a single search step in the algorithm: evaluate removing a single feature from
        the feature set and choose the one with better error after testing all of them
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :return: the best feature set found and its error
        """
        best_feature = None
        best_err = float('inf')
        for i in self.vars:
            err = self.eval_feature(df, obj_var, i)
            if err < best_err:
                best_err = err
                best_feature = i

        return best_feature, best_err

    def eval_feature(self, df, obj_var, col):
        """
        Evaluate the performance of removing a feature from the feature set
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param col: the feature to evaluate
        :return: the average error obtained from evaluating the cross-validation
        """
        m = self.model(components=self.components)
        if col is None:
            feature_var = self.vars
        else:
            feature_var = self.vars.difference([col])
        err = m.eval_cv(df, obj_var, feature_var, self.metrics,
                        self.folds, self.shuffle, self.seed, print_res=False)

        return err
