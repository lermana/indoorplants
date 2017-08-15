import numpy as np
import pandas as pd

import evaluation


def balance_binary_classes(y):
    """Passed a Pandas Series class column, returns List
    of balanced indices, with each set of indices providing
    a roughly 50/50 mix of positive & negative class instances.
    
    Length of list is determined by ratio of major (more 
    prevalent) to minor (less prevalent) class instances."""
    val_counts = y.value_counts()
    get_counts = lambda _: val_counts[_]
    
    cnt_by_cnt = {get_counts(_): _ for _ in (0, 1)}
    cnt_by_cls = {_: get_counts(_) for _ in (0, 1)}
    
    major = cnt_by_cnt[max(cnt_by_cnt)]
    minor = cnt_by_cnt[min(cnt_by_cnt)]
    
    ratio = round(cnt_by_cls[major]/ cnt_by_cls[minor])
    minor_inds = y[y == minor].index.values
    avail_inds = y[y == major].index.values
    i, bal_inds = 1, []
    
    while i < ratio:
        next_set = np.random.choice(avail_inds,
                                    size=len(minor_inds),
                                    replace=False)
        bal_inds.append(np.concatenate((minor_inds,
                                        next_set)))
        avail_inds = np.setdiff1d(avail_inds, next_set)
        i += 1
    
    bal_inds.append(np.concatenate((minor_inds,
                                    avail_inds)))
    return bal_inds


def _cv_format(X, y, model, bal_inds, score_funcs, 
              splits=5, scale_obj=None, train_scores=True):
    """Return DataFrame of scores in 'score_funcs' for each fold
    of cross validation across all sets of balanced indices, 
    where each set of indices has its own splits-fold CV 
    performed."""
    i, num, results = 0, len(bal_inds), []
    while i < num:
        results.append(
            pd.concat({i + 1: 
                evaluation._cv_format(X.loc[bal_inds[i], :], 
                                      y.loc[bal_inds[i]], 
                                      model, score_funcs, 
                                      splits, scale_obj, 
                                      train_scores)}))
        i += 1
    return pd.concat(results)


def cv_score(X, y, model, bal_inds, score_funcs, 
             splits=5, scale_obj=None, train_scores=True):
    """For each set of indices in bal_inds, performs
    'splits'-fold cross validation of 'model' and calculates each 
    score passed in 'score_funcs', returning descriptive
    statistics (calculated across all trials for all indices)
    are then returned as a DataFrame.

    If 'scale_obj' is passed, 'X' will be scaled within
    each fold so as to prevent data leakage.

    'train_scores' [default True] specifies reporting 
    on train scores."""
    return evaluation._cv_score(_cv_format(X, y, model, bal_inds, 
                                           score_funcs, splits, 
                                           scale_obj, train_scores))


def cv_conf_mat(X, y, model, bal_inds, splits=5, 
                scale_obj=None,):
    """Returns confusion matrix for each cross validation
    trial for each index set."""
    i, num, results = 0, len(bal_inds), []
    while i < num:
        df = table[table.index.isin(bal_inds[i])]
        results.append(
            pd.concat({i + 1: 
                evaluation.cv_conf_mat(df[X_cols], df[y_cols],
                                       model, splits, scale_obj)}))
        i += 1
    return pd.concat(results)
