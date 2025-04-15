from mlworkflow.utils import print_cond
from mlworkflow.models.model import check_model
from mlworkflow.models.metrics import check_metrics
from mlworkflow.sanitation import check_pd, check_strs, check_vars
from tqdm import tqdm
import numpy as np


class GeneticVarSel:
    """
    Genetic algorithm for variable selection. Binary version for adding/removing a variable from the feature set.
    The algorithm uses cross-validation to test the different feature sets.
    """
    def __init__(self, model, metrics, components=[], max_features=20, max_it=50, n_inds=20,
                 per_cross=0.5, per_mut=0.05, seed=None):
        check_model(model)
        check_metrics(metrics)

        self.model = model
        self.metrics = metrics
        self.best_err = float('inf')
        self.components = components
        self.max_features = max_features
        self.max_it = max_it
        self.n_inds = n_inds
        self.per_cross = per_cross
        self.per_mut = per_mut
        self.rng = np.random.default_rng(seed=seed)
        self.best_set = None
        self.features = None
        self.inds = None
        self.inds_scr = None
        self.inds_rank = None
        self.folds = None
        self.shuffle = None
        self.seed = None

    def run(self, df, obj_var, ini_feat=[], folds=5, shuffle=False, seed=None, print_res=True):
        """
        Run the search algorithm to find the best variable set possible and its error
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

        # Initialize
        self.features = df.drop(obj_var, axis='columns').columns
        self.initialize_pop()
        self.eval_inds(df, obj_var)
        if len(ini_feat) > 0:
            # self.set_features(ini_feat)  # Poor performance
            self.insert_initial_inds(ini_feat)
            self.eval_inds(df, obj_var)

        for _ in tqdm(range(self.max_it)):
            print_cond(print_res, f"\nCurrent feature set: {list(self.features[self.best_set > 0])}")
            print_cond(print_res, f"Current best error: {self.best_err}")
            # Crossover
            new_inds = self.crossover()
            # Mutation of the new individuals
            self.mutate(new_inds)
            # Replacement
            self.replace(new_inds)
            # Evaluation
            self.eval_inds(df, obj_var)

        print_cond(print_res, f"Best feature set found: {list(self.features[self.best_set > 0])}")

        return self.features[self.best_set > 0], self.best_err

    def to_str(self):
        return 'Genetic algorithm characteristics:\n' + str(self.__dict__)

    def initialize_pop(self):
        """
        Initialize the population. This generates n_inds binary numpy arrays. A 1 means that feature is selected and
        a 0 means it is not selected. Kept simple for now
        """
        self.inds = np.array([self.rng.choice(2, len(self.features)) for _ in range(self.n_inds)])

    def one_point_cross(self, par1, par2):
        """
        Simple random one point crossover between two individuals
        """
        point = self.rng.choice(len(par1), 1)[0]
        res = np.concatenate((par1[0:point], par2[point:len(par2)]))
        return res

    def crossover(self):
        """
        Crossover operation over a percentage of the individuals
        """
        n_new_inds = round(self.n_inds * self.per_cross)
        new_inds = np.empty((n_new_inds, len(self.features)), dtype='int')

        for i in range(n_new_inds):
            idx = self.rng.choice(self.n_inds, 2, replace=False)
            new_inds[i] = self.one_point_cross(self.inds[idx[0]], self.inds[idx[1]])

        return new_inds

    @staticmethod
    def flip_bit(bit):
        res = 1
        if bit > 0:
            res = 0

        return res

    def bitwise_mutation(self, ind):
        """
        Bitwise mutation performed by reference on the original individual. It flips bits that are sampled to lower
        than per_mut in a [0,1] uniform distribution, so bits have per_mut chance to be flipped
        """
        mutation = self.rng.uniform(0, 1, len(ind)) < self.per_mut
        ind[mutation] = [self.flip_bit(x) for x in ind[mutation]]

    def mutate(self, inds):
        """
        Mutation operator for a set of individuals.
        """
        for ind in inds:
            self.bitwise_mutation(ind)

    def replace(self, new_inds):
        """
        Replace the worst ranking individuals from the population with the new individuals new_inds
        """
        idx_worst = self.inds_rank[-len(new_inds):]
        self.inds[idx_worst] = new_inds

    def eval_inds(self, df, obj_var):
        """
        Evaluate all individuals in the current population
        """
        self.inds_scr = np.array([self.eval_bin_set(df, obj_var, bin_set) for bin_set in self.inds])
        self.inds_rank = np.argsort(self.inds_scr)
        if self.inds_scr[self.inds_rank[0]] < self.best_err:
            self.best_err = self.inds_scr[self.inds_rank[0]]
            self.best_set = self.inds[self.inds_rank[0]]

    def eval_bin_set(self, df, obj_var, bin_set):
        """
        Evaluate the performance of a feature set
        :param df: the pandas dataframe with the data
        :param obj_var: a list with the name of the objective variable
        :param bin_set: the binary feature set to be evaluated
        :return: the average error obtained from evaluating the cross-validation
        """
        m = self.model(components=self.components)
        feature_set = self.features[bin_set > 0]
        err = m.eval_cv(df, obj_var, feature_set, self.metrics,
                        self.folds, self.shuffle, self.seed, print_res=False)

        return err

    def translate(self, feat_set):
        return np.in1d(self.features, feat_set).astype(int)

    def set_features(self, ini_feat):
        """
        Set the features in ini_feat to 1 in all the individuals of the population. Poor performance overall
        """
        mask = self.translate(ini_feat)
        self.inds = np.bitwise_or(self.inds, mask)

    def insert_initial_inds(self, ini_feat, n=4):
        """
        Insert n individuals with the initial feature set proposed into the population
        """
        ind = self.translate(ini_feat)
        new_inds = np.stack([ind]*n)
        self.replace(new_inds)


