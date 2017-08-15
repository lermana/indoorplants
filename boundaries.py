import numpy as np
import pandas as pd
from functools import partial, update_wrapper

import evaluation
import balance


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


def cv_score(X, y, model, score_funcs, thresholds, 
             splits=5, bal_inds=None, train_scores=True, 
             scale_obj=None):
    """Performs cross validation of passed 'model' for the
    different decision boundaries passed in 'thresholds',
    which must be an iterable.

    Returns balance.cv_score if 'bal_inds' else 
    evaluation.cv_score. Please see these functions' docs 
    for more information on other arguments."""
    model.predict = model.predict_proba
    if bal_inds is None: cv_func = evaluation.cv_score
    else: cv_func = partial(balance.cv_score, 
                            bal_inds=bal_inds)

    results = {}
    for t in thresholds:
        scores = _convert_score_funcs(t, score_funcs)
        kwargs = {'X':X, 'y':y, 'model':model, 'splits':splits,
                  'scale_obj':scale_obj, 'score_funcs':scores}
        results[t] = cv_func(**kwargs)

    return pd.concat(results)