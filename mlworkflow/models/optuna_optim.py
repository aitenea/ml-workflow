from mlworkflow.models.metrics import check_metrics
from mlworkflow.models.model import check_model_instance


class OptunaOptim:
    def __init__(self, model, metrics, bounds, n_trials=100, direction='minimize'):
        check_model_instance(model)
        check_metrics(metrics)
        self.model = model
        self.metrics = metrics
        self.bounds = bounds
        self.n_trials = n_trials
        self.direction = direction

    def run(self, df, obj_var, feature_var, folds=5,
            shuffle=False, seed=None, splits=None, print_res=True):
        try:
            import optuna
        except ImportError:
            raise ImportError('The Optuna module needs to be installed for this parameter tuning.')
        study = optuna.create_study(direction=self.direction)
        study.optimize(lambda trial: self.objective(trial, self.model, self.bounds, df, obj_var, feature_var,
                                                    self.metrics, folds, shuffle=shuffle, seed=seed, splits=splits,
                                                    print_res=print_res), n_trials=self.n_trials)

        return study.best_params

    def to_str(self):
        return 'Optuna study characteristics:\n' + str(self.__dict__)

    @staticmethod
    def objective(trial, model, bounds, df, obj_var, feature_var, metrics, folds, shuffle, seed, splits, print_res):
        params = []

        for i in range(len(bounds)):
            params.append(trial.suggest_float(f'Param_{i}', bounds[i][0], bounds[i][1], log=True))

        model.assign_params(params)
        res, preds = model.eval_cv(df, obj_var, feature_var, metrics, folds=folds,
                                   shuffle=shuffle, seed=seed, splits=splits, print_res=print_res)

        return res
