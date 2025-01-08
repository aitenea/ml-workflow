from scipy.optimize import differential_evolution
from scipy.stats import qmc
from mlworkflow.models.metrics import check_metrics
from mlworkflow.models.model import check_model_instance


class DiffEvOptim:
    def __init__(self, model, metrics, bounds, maxiter=100, popsize=15):
        check_model_instance(model)
        check_metrics(metrics)
        self.model = model
        self.metrics = metrics
        self.bounds = bounds
        self.maxiter = maxiter
        self.popsize = popsize

    def run(self, df, obj_var, feature_var, folds=5, init='latinhypercube',
            shuffle=False, seed=None, splits=None, print_res=True):
        if not isinstance(init, str):
            init = self.sample_initial_pop(init)
        res = differential_evolution(self._optim_model, self.bounds, maxiter=self.maxiter,
                                     popsize=self.popsize, init=init, workers=1,
                                     args=(df, self.model, self.metrics, obj_var, feature_var, folds, shuffle, seed, splits, print_res))

        return res

    def to_str(self):
        return 'Differential evolution characteristics:\n' + str(self.__dict__)

    @staticmethod
    def _optim_model(x, df, model, metrics, obj_var, feature_var, folds, shuffle, seed, splits, print_res):
        """
        Function to be passed onto the optimizer. It takes a parameter vector x and returns the error from the
        Metrics object to be minimized
        :param x: a list of values to be assigned as parameters
        """
        model.assign_params(x)
        model.fit(df, obj_var, feature_var)
        res = model.eval_cv(df, obj_var, feature_var, metrics, folds=folds,
                            shuffle=shuffle, seed=seed, splits=splits, print_res=print_res)

        return res

    def sample_initial_pop(self, ini_inds):
        sampler = qmc.LatinHypercube(d=len(self.bounds))
        pop = sampler.random(self.popsize)
        l_bounds = list(map(lambda x: x[0], self.bounds))
        u_bounds = list(map(lambda x: x[1], self.bounds))
        pop_scaled = qmc.scale(pop, l_bounds, u_bounds)
        pop_scaled[0:len(ini_inds)] = ini_inds

        return pop_scaled
