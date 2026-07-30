from mlworkflow.utils import print_cond
from mlworkflow.models.model import check_model
from mlworkflow.models.metrics import check_metrics
from mlworkflow.sanitation import check_pd, check_strs, check_vars
import bisect


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


class BeamSearchVarSel:
    """
    Beam search wrapper algorithm for variable selection. The algorithm uses cross-validation to test the different
    feature sets, so it can arrive to different results from the same starting point. By default, the cross-validation
    is deterministic without shuffle, and so is the algorithm
    """
    def __init__(self, model, metrics, components=[], w=4, b=None, max_features=20, force=False):
        check_model(model)
        check_metrics(metrics)

        self.model = model
        self.metrics = metrics
        self.w = w
        self.b = b
        self.vars = [[] for _ in range(self.w)]
        self.best_err = [float('inf') for _ in range(self.w)]
        self.max_features = max_features
        self.force = force
        self.components = components
        self.folds = None
        self.shuffle = None
        self.seed = None

    def run(self, df, obj_var, ini_feat=[], folds=5, shuffle=False, seed=None, print_res=True):
        """
        Run the feature subset selection algorithm
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
            self.vars = [ini_feat] * self.w
            self.best_err = [self.eval_feature(df, obj_var, None, group=x) for x in range(self.w)]

        features_res = self.initial_search_step(df, obj_var)
        for i in range(self.w):
            self.vars[i].append(features_res[i][1])
            self.best_err[i] = features_res[i][0]
        end = False

        while not end and max([len(x) for x in self.vars]) < self.max_features:
            print_cond(print_res, f"Current feature sets: {self.vars}")
            print_cond(print_res, f"Current best errors: {self.best_err}")
            features_res = self.search_step(df, obj_var)
            print_cond(print_res, f"Best features found: {[x[1] for x in features_res]}, "
                                  f"with error {[x[0] for x in features_res]}")
            if min(features_res)[0] < min(self.best_err) or self.force:
                tmp = []
                for feat in features_res:
                    tmp.append(self.vars[feat[2]] + [feat[1]])
                self.vars = tmp
                self.best_err = [x[0] for x in features_res]
            else:
                end = True

        print_cond(print_res, f"Best feature set found: {self.vars[0]}")

        return self.vars, self.best_err

    def to_str(self):
        return 'Beam search algorithm characteristics:\n' + str(self.__dict__)

    def search_step(self, df, obj_var):
        """
        Execute a single search step in the algorithm: evaluate in each possibility of the beam search adding a single
        feature from all possible ones to the feature set and choose the w ones with better error after testing all of
        them
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :return: the best w feature sets found and their error
        """
        res = [(float('inf'), None, None)] * self.w  # Error, feature, group

        for g in range(self.w):
            idx = df.columns.difference(self.vars[g] + obj_var)
            for i in idx:
                err = self.eval_feature(df, obj_var, i, g)
                if err < max(res)[0]:
                    bisect.insort(res, (err, i, g))
                    res = res[0:self.w]

        return res

    def initial_search_step(self, df, obj_var):
        """
        First single search step in the algorithm. Single pass to populate the initial tree branching into different
        scenarios
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :return: the best w feature sets found and their error
        """
        res = [(float('inf'), None, None)] * self.w  # Error, feature, group

        idx = df.columns.difference(self.vars[0] + obj_var)
        for i in idx:
            err = self.eval_feature(df, obj_var, i, 0)
            if err < max(res)[0]:
                bisect.insort(res, (err, i, 0))
                res = res[0:self.w]

        return res

    def eval_feature(self, df, obj_var, col, group):
        """
        Evaluate the performance of adding a feature to the feature set of a specific branch in the beam search
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param col: the feature to evaluate
        :param group: the "tree branch" to add the feature to
        :return: the average error obtained from evaluating the cross-validation
        """
        m = self.model(components=self.components)
        if col is None:
            feature_var = self.vars[group]
        else:
            feature_var = self.vars[group] + [col]
        err = m.eval_cv(df, obj_var, feature_var, self.metrics,
                        self.folds, self.shuffle, self.seed, print_res=False)

        return err