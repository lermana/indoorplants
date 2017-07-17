import numpy as np
import pandas as pd

import evaluation


def balance_binary_classes(y):
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


def _cv(table, bal_inds, X_cols, y_cols, model, score_func, 
        splits=5, scale_obj=None):
    i, num, results = 0, len(bal_inds), []
    while i < num:
        new_table = table[table.index.isin(bal_inds[i])]
        results.append(
            pd.concat({i + 1: 
                evaluation._cv(new_table[X_cols], 
                               new_table[y_cols],
                               model, score_func, splits,
                               scale_obj)}))
        i += 1
    return pd.concat(results)


def cv_score(table, bal_inds, X_cols, y_cols, model, score_func, 
             splits=5, scale_obj=None):
    results = _cv(table, bal_inds, X_cols, y_cols, model, 
                  score_func, splits, scale_obj)
    m = results.groupby(results.index.get_level_values(1)).mean()
    s = results.groupby(results.index.get_level_values(1)).std()
    return pd.concat({'mean': m}, axis=1
                ).join(pd.concat({'std. dev.': s}, axis=1))