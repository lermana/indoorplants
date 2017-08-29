import numpy as np
import pandas as pd
from itertools import chain

import validation


def _cv_proba(X, y, model_obj, splits=5, scale_obj=None):

    model_obj.predict = model_obj.predict_proba
    score = lambda t, p: np.vstack([t.values, p[:, 1]]).T
    return validation._cv_engine(X, y, model_obj,
                [score], splits, scale_obj, False)
                         

def cv_rank(X, y, model_obj, splits=5, scale_obj=None):

    results = pd.DataFrame(np.vstack(chain(
        *_cv_proba(X, y, model_obj, splits, scale_obj))),
        columns=['class', 'proba'])

    grouped = results[['class', 'proba']
                         ].groupby(['class']
                         ).median().rename(
                         columns={'proba':'median'})
    return grouped.join(
                results[['class', 'proba']
                           ].groupby(['class']
                           ).mad().rename(
                            columns={'proba':'mad'}))


def cv_calibrate(X, y, model_obj, splits=5, scale_obj=None,
                 pos_only=True):

    results = _cv_proba(X, y, model_obj, splits, scale_obj)
    df = pd.concat({i: pd.DataFrame(data)for i, data in 
                        enumerate(list(chain(*results)))})
    df.columns = ['class', 'proba']
    df['prob_bin'] = df.proba.round(1)
    
    grouped = df[['class', 'prob_bin']
                   ].groupby([df.index.get_level_values(0),
                                 'class', 'prob_bin']
                   ).size().to_frame().rename(
                        columns={0: 'count'})
    grouped = grouped.reset_index(
                    ).merge(grouped.groupby(level=[0, 2]
                                  ).sum().reset_index(
                                  ).rename(columns={'count': '_'}),
                              on=['level_0', 'prob_bin']
                     ).set_index(['level_0', 'class', 'prob_bin'])
    grouped['pcnt'] = grouped['count'] / grouped['_']

    final = grouped.unstack(level=1).loc[:, 'pcnt'].fillna(0)
    final = pd.concat(
                {'mean': final.groupby(level=[1]).mean()},
                    axis=1).join(pd.concat(
                        {'std': final.groupby(level=[1]).std()},
                             axis=1)).swaplevel(0, 1, 1)
    if pos_only is True: return final.loc[:, 1.0]
    else: return final