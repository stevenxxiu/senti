
import jsonpickle
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.externals import joblib

# XXX actually we can create some wrappers to individual classes and re-use Memory.cache, inputs can be specified via
# XXX an input_id argument


class PersistPipeline(Pipeline):
    def get_params(self, deep=True, models=True):
        params = super().get_params(deep=deep)
        if not models:
            params = dict((key, value) for key, value in params.items() if '__' in key)
        return params

    def _step_params(self, params):
        res = dict((step, {}) for step, _ in self.steps)
        for pname, pval in params.items():
            step, param = pname.split('__', 1)
            res[step][param] = pval
        return res

    def _find_first_recalc_step(self, input_id, fit_params, prev_input_id, prev_params, prev_fit_params):
        params = self._step_params(self.get_params(models=False))
        fit_params = self._step_params(fit_params)
        prev_params = self._step_params(prev_params)
        prev_fit_params = self._step_params(prev_fit_params)
        if input_id != prev_input_id:
            return 0
        for i, step in enumerate(self.steps):
            if params[step[0]] != prev_params[step[0]] or fit_params[step[0]] != prev_fit_params[step[0]]:
                return i
        return len(self.steps)

    def _pre_transform(self, X, y=None, **fit_params):
        pass
    
    def fit(self, X, y=None, input_id=None, prev_input_id=None, prev_params=None, prev_fit_params=None, **fit_params):
        first_fit = prev_params is None and prev_fit_params is None
        if first_fit:
            with open('fit.options.json') as sr:
                obj = jsonpickle.decode(sr.read())
                prev_input_id, prev_params, prev_fit_params = obj['input_id'], obj['params'], obj['fit_params']
        i = self._find_first_recalc_step(input_id, prev_input_id, fit_params, prev_params, prev_fit_params)
        if i != 0:
            # load the latest model
            self.steps[i - 1][1] = joblib.load('{}.joblib'.format(self.steps[i - 1][0]))
        for step in self.steps[i:]:
            # XXX store these models, ignore PersistFeatureUnion since it stores itself

        if first_fit:
            with open('fit.options.json', 'w') as sr:
                sr.write(jsonpickle.encode({
                    'input_id': input_id, 'params': self.get_params(models=False), 'fit_params': fit_params
                }))

    def transform(self, X, input_id=None, prev_input_id=None, prev_params=None):
        # XXX more models need to be loaded if the input's different
        pass


class PersistFeatureUnion(FeatureUnion):
    pass
