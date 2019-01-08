import numpy as np
import pandas as pd
from functools import partial, update_wrapper
from sklearn.metrics import confusion_matrix

from indoorplants.validation import crossvalidate


def _score_prob_as_class(t, func):
    """Returns modified 'func' that is ready to take
    a vector of class probabilities and return a 
    the results of a score function that relies on
    class labels."""
    def threshold(t, y_true, y_pred):
        pos_class = y_pred[:, 1]
        conv = np.vectorize(lambda _: 1 if _ > t else 0)
        return func(y_true, conv(pos_class))

    def wrap(t):    
        partial_func = partial(threshold, t)
        update_wrapper(partial_func, func)
        return partial_func
    
    return wrap(t)


def _convert_score_funcs(t, score_funcs):
    """Takes an iterable of scoring functions and returns
    modified version of each function, modified by 
    _score_prob_as_class."""
    new_funcs = []
    for score in score_funcs:
        new_funcs.append(_score_prob_as_class(t, score))
    return new_funcs


def cv_score(X, y, model_obj, score_funcs, thresholds, 
             splits=5, train_scores=True, scale_obj=None):
    """Performs cross validation of passed 'model' for the
    different decision boundaries passed in 'thresholds',
    which must be an iterable.

    Returns crossvalidate.cv_score. Please see this function's 
    docs for more information on other arguments."""
    model_obj.predict = model_obj.predict_proba
    results = {}
    for t in thresholds:
        scores = _convert_score_funcs(t, score_funcs)
        kwargs = {'X':X, 'y':y, 'model_obj':model_obj, 
                  'splits':splits, 'scale_obj':scale_obj, 
                  'score_funcs':scores, 
                  'train_scores': train_scores}
        results[t] = crossvalidate.cv_score(**kwargs)
    return pd.concat(results)


def cv_conf_mat(X, y, model_obj, t, splits=3, scale_obj=None):
    """Returns confusion matrix resulting from 'splits'-fold
    cross-validation using 't' to determine decision boundary."""
    model_obj.predict = model_obj.predict_proba
    scores = _convert_score_funcs(t, [confusion_matrix])
    
    results = crossvalidate.cv_engine(X=X, y=y, model_obj=model_obj, 
                                     score_funcs=scores, 
                                     splits=splits, scale_obj=scale_obj, 
                                     train_scores=False) 
    results = [pd.concat({i: pd.DataFrame(trial[0],
                            index=['neg_true', 'pos_true'],
                            columns=['neg_pred', 'pos_pred'])}) 
                   for i, trial in enumerate(results, 1)]

    return pd.concat({t: pd.concat(results)})