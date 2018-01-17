import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score

import sklearn.model_selection as skl
import matplotlib.pyplot as plt

from indoorplants.validation import crossvalidate, \
                                    calibration, \
                                    boundaries


def _validate_param_range(model_type, param_name, param_range,
                          X, y, score_funcs, other_params={},
                          splits=5, scale_obj=None, train_scores=True):
    """Returns validation.cv_score across values in 'param_range'
    for 'param_name', which should be a working parameter for the
    passed model.

    'model_type' should be an uninstantiated sklearn model (or
    one with similar fit and predict methods). Additional 
    hyper-parameters (i.e. not 'param_name' should be passed
    in to 'other_params' as dictionary.

    Please see validation.cv_score for details on other args.""" 
    results = {}
    for val in param_range:
        model_obj = model_type(**{param_name:val}, **other_params)
        
        kwargs = {'model_obj': model_obj, 'X': X, 'y': y, 
                  'splits': splits, 'scale_obj': scale_obj, 
                  'train_scores': train_scores, 
                  'score_funcs': score_funcs}
        try:
            _ = val / 1
        except TypeError:
            val = str(val)
        results[val] = crossvalidate._cv_format(**kwargs)
    return pd.concat(results)


def validation_curve(X, y, score, model_type, param_name, 
                     param_range, other_params={}, splits=5, 
                     scale_obj=None, semilog=False, figsize=(11, 8)):
    """Cross validates 'model_type' across passed parameters and
    plots results. Please see _validate_param_range for more
    details around the cross validation arguments.
    
    Pass True for 'semilog' if 'param_range' values would be better
    visualized with log scaling. Pass tuple to 'figsize' if you 
    wish to override default of (11, 8)."""

    results = _validate_param_range(model_type, param_name, 
                                    param_range, X, y, [score], 
                                    other_params, splits, scale_obj)
    
    means = results.groupby(level=0).mean().reset_index()
    stds = results.groupby(level=0).std().reset_index()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(means.index)

    if semilog is True: plt_func = plt.semilogx
    else: plt_func = plt.plot
    plt_func(means.index, means[(score.__name__, 'train')
                                ].values, 
             label='train', color='darkorange', lw=2)
    plt_func(means.index, means[(score.__name__, 'test')
                                ].values, 
             label='validation', color="navy", lw=2)

    bands = lambda _: (means[(score.__name__, _)]
                        - stds[(score.__name__, _)],
                       means[(score.__name__, _)]
                        + stds[(score.__name__, _)])
    plt.fill_between(means.index, *bands('train'), 
                     alpha=0.1, color="darkorange", lw=2)
    plt.fill_between(means.index, *bands('test'), 
                     alpha=0.1, color="navy", lw=2)

    xlab = ax.set_xlabel(param_name)
    xlab = ax.set_xticklabels(means['index'].values)
    ylab = ax.set_ylabel(score.__name__)
    title = plt.title('Validation curve: {}, across {}'.format(
                      model_type.__name__, param_name))
    plt.legend(loc="best")


def _validate_train_sizes(X, y, model_obj, score_funcs,  
                         scale_obj=None, splits=5,
                         train_sizes=[0.1, 0.33, 0.55, 0.78, 1.]):
    """Returns validation.cv_score for 
    for 'param_name', which should be a working parameter for the
    passed model.

    'model_type' should be an uninstantiated sklearn model (or
    one with similar fit and predict methods). Additional 
    hyper-parameters (i.e. not 'param_name' should be passed
    in to 'other_params' as dictionary.

    Please see validation.cv_score for details on other args.""" 
    val_counts = y.value_counts()
    get_counts = lambda _: val_counts[_]
    
    cnt_by_cnt = {get_counts(_): _ for _ in (0, 1)}
    cnt_by_cls = {_: get_counts(_) for _ in (0, 1)}
    
    major = cnt_by_cnt[max(cnt_by_cnt)]
    minor = cnt_by_cnt[min(cnt_by_cnt)]
    
    ratio = cnt_by_cls[minor] / len(y)
    results = {}
    for val in train_sizes:
        size = int(np.floor(len(y) * val))
        min_size = int(np.floor(size * ratio))
        maj_size = size - min_size
    
        y_sized = y[y==major].sample(maj_size
                    ).append(y[y==minor].sample(min_size))
        X_sized = X.loc[y_sized.index, :]

        kwargs = {'model_obj': model_obj, 'X': X_sized, 
                  'y': y_sized, 'splits': splits, 
                  'scale_obj': scale_obj, 
                  'train_scores': True, 
                  'score_funcs': score_funcs}
        
        results[str(val)] = crossvalidate._cv_format(**kwargs)
    return pd.concat(results)


def learning_curve(X, y, model_type, score, scale_obj=None, 
                   splits=5, train_sizes=[0.1, 0.33, 0.55, 0.78, 1.],
                   model_params={}, figsize=(11, 8)):
    """Cross validates 'model_type' across passed parameters and
    plots results. Please see _validate_param_range for more
    details around the cross validation arguments.
    
    Pass True for 'semilog' if 'param_range' values would be better
    visualized with log scaling. Pass tuple to 'figsize' if you 
    wish to override default of (11, 8)."""
    model_obj = model_type(**model_params)
    results = _validate_train_sizes(X, y, model_obj, [score], 
                                    scale_obj, splits, train_sizes)
    
    means = results.groupby(level=0).mean().reset_index()
    stds = results.groupby(level=0).std().reset_index()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(means.index)

    plt.plot(means.index, means[(score.__name__, 'train')
                                ].values, 
             label='train', color='crimson', lw=2)
    plt.plot(means.index, means[(score.__name__, 'test')
                                ].values, 
             label='validation', color="teal", lw=2)

    bands = lambda _: (means[(score.__name__, _)]
                        - stds[(score.__name__, _)],
                       means[(score.__name__, _)]
                        + stds[(score.__name__, _)])

    plt.fill_between(means.index, *bands('train'), 
                     alpha=0.1, color="crimson", lw=2)
    plt.fill_between(means.index, *bands('test'), 
                     alpha=0.1, color="teal", lw=2)

    xlab = ax.set_xlabel('training size')
    xlab = means['index'].map(lambda _: int(round(float(_) * len(y), -2)))
    xlab = ax.set_xticklabels(xlab)
    ylab = ax.set_ylabel(score.__name__)
    title = plt.title('Learning curve: {}'.format(
                                model_type.__name__))
    plt.legend(loc="best")


def calibration_curve(X, y, model_type, scale_obj=None, splits=5, 
                      model_params={}, calib_types=None, figsize=(11, 8)):
    """Plots calibration curves for original model & passed calibrators."""
    def plot_probs(results, c, label):
        plt.plot(results.index, results['mean'],
                 label=label, color=c, lw=2)
        bands = lambda _: (_['mean'] - _['std'],
                           _['mean'] + _['std'])
        plt.fill_between(results.index, *bands(results), 
                         alpha=0.1, color=c, lw=2)

    model_obj = model_type(**model_params)
    results = calibration.cv_calibrate(X, y, model_obj, splits, 
                                       scale_obj, calib_types)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(results.index)
    colors = (c for c in ['C6', 'C4', 'C0'])

    plot = plot_probs(results.loc[:, 'original_model'], 
                      colors.__next__(), 
                      'original_model')

    if calib_types is not None:
        for cal_mod in calib_types:
            plot = plot_probs(results.loc[:, cal_mod.__name__ + '_cal'], 
                              colors.__next__(), 
                              cal_mod.__name__)

    ax.set_xlim(0, 1)
    xlab = ax.set_xlabel('predicted probability (bin)')
    ax.set_ylim(0, 1)
    ylab = ax.set_ylabel('% of data with positive label')
    title = plt.title('Calibration curve: {}'.format(
                                model_type.__name__))
    plt.legend(loc="best")



def precision_recall_curve(X, y, model_type, scale_obj=None, 
                           splits=5, model_params={}, figsize=(11, 8),
                           thresholds=[.1*x for x in range(1, 10)]):
    """Plot precision-recall curve over decision boundaries:
    [0, 1] for binary classification. 

    Pass 'model_type', 'model_params' and 'figsze' in same fashion 
    as for learning_curve. 'splits' and 'scale_obj' are same as
    for all cv_score functions. """
    results = boundaries.cv_score(X, y, model_type(**model_params), 
                                 [recall_score, precision_score],
                                 thresholds,
                                 splits, False, scale_obj)
    to_plot = results.unstack()['mean']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(to_plot['recall_score'], to_plot['precision_score'], lw=2)
    plt.fill_between(to_plot['recall_score'], to_plot['precision_score'],
                     alpha=.2)

    for row in to_plot.itertuples():
        ax.annotate('{}'.format(round(row[0], 3)),
                    xy=(row[1], row[2]),
                    xytext=(row[1] - .01, 
                            row[2] + .02))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title = plt.title('Precision & recall by decision boundary: {}'.format(
                                        model_type.__name__))