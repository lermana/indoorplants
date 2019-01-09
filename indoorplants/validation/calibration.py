import numpy as np
import pandas as pd
from itertools import chain

from indoorplants.validation import crossvalidate


def _cv_proba(X, y, model_obj, splits=5, scale_obj=None):
    """Return numpy array of 'splits'-fold CV results, 
    where results are:
    -column 0: actual class
    -columns 1 & 2: proba. of each of neg. and pos. classes."""
    model_obj.predict = model_obj.predict_proba
    score = lambda t, p: np.hstack([t.values.reshape(-1, 1), 
                                    p[:, 1].reshape(-1, 1)])
    return crossvalidate.cv_engine(X, y, model_obj,
                [score], splits, scale_obj, False)
                         

def _get_rank_stats(results):
    """Helper function that produces median and m.a.d
    probabilities for each of 2 classes given _cv_proba 
    results with reworked columns."""
    grouped = results[['class', 'proba']
                         ].groupby(['class']
                         ).median().rename(
                         columns={'proba':'median'})
    return grouped.join(
                results[['class', 'proba']
                           ].groupby(['class']
                           ).mad().rename(
                            columns={'proba':'mad'}))


def cv_rank(X, y, model_obj, splits=5, scale_obj=None):
    """Returns median and m.a.d. probabilities for each class
    for test results over 'splits'-fold CV."""
    results = pd.DataFrame(np.vstack(chain(
        *_cv_proba(X, y, model_obj, splits, scale_obj))),
        columns=['class', 'proba'])
    return _get_rank_stats(results)


def _calibrate_cv(model_results, calib_type, splits):
    """Produces calibrated probabilities for passed 'model_results'
    using passed 'calib_type' calibration model type 
    (uninstantiated model object)."""
    calib_res = [np.hstack(
                    [model_results.loc[i, 'class'
                            ].values.reshape(-1, 1),
                     calib_type(
                        ).fit(
                            model_results.loc[i, 'proba'
                            ].to_frame(),
                            model_results.loc[i, 'class']
                        ).predict_proba(
                            model_results.loc[i, 'proba'
                            ].to_frame()
                            )[:, 1].reshape(-1, 1)])
                 for i in range(splits)]
    df = pd.concat({i: pd.DataFrame(data) for 
                        i, data in enumerate(calib_res)})
    df.columns = ['class', 'proba']
    return df


def _cv_calibrate(X, y, model_obj, splits=5, scale_obj=None,
                  calib_types=None):
    """Returns CV results for passed model, and calibrates results."""
    model_res = _cv_proba(X, y, model_obj, splits, scale_obj)
    df = pd.concat({i: pd.DataFrame(data)for i, data in 
                        enumerate(chain(*model_res))})
    df.columns = ['class', 'proba']
    if calib_types is None: return pd.concat({'original_model': df}, axis=1)

    dfs_cal = pd.concat(
                [pd.concat({cal_type.__name__ + '_cal': 
                    _calibrate_cv(df, cal_type, splits)},
                 axis=1) for cal_type in calib_types])
    return pd.concat({'original_model': df}, axis=1).join(dfs_cal)


def _prob_bin_stats(results, pos_only=True):
    """Bins passed probabilities and calculates mean and std. 
    actual positive class frequencies."""
    df = results.copy()
    df['prob_bin'] = df.proba.round(1)
    
    grouped = df[['class', 'prob_bin']
                 ].groupby([df.index.get_level_values(0),
                            'class', 'prob_bin']
                 ).size().to_frame().rename(columns={0: 'count'})

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


def cv_calibrate(X, y, model_obj, splits=5, scale_obj=None,
                 calib_types=None, pos_only=True):
    """Return mean and std. CV results comparing actual positive
    class probabilities to binned predicted probabilities, for 
    the original model and all passed calibrators."""
    res = _cv_calibrate(X, y, model_obj, splits, scale_obj,
                           calib_types)
    modeled = pd.concat({'original_model':
                _prob_bin_stats(res.loc[:, 'original_model'])}, axis=1)
    if calib_types is None: return modeled

    calib_results = [pd.concat({mod.__name__ + '_cal': 
                        _prob_bin_stats(res.loc[:, mod.__name__ + '_cal'])},
                        axis=1) for mod in calib_types]
    return modeled.join(pd.concat(calib_results))