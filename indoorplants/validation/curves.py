import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score

import sklearn.model_selection as skl
import matplotlib.pyplot as plt

from indoorplants.validation import crossvalidate, \
                                    calibration, \
                                    boundaries


def validation_curve(X, y, score, model_type, param_name, 
                     param_range, other_params={}, splits=5, 
                     scale_obj=None, semilog=False, figsize=(11, 8)):
    """
    Cross validates `model_type` across passed parameters and
    plots results. Please see _validate_param_range for more
    details around the cross validation arguments.
    
    Pass True for `semilog` if `param_range` values would be better
    visualized with log scaling. Pass tuple to `figsize` if you 
    wish to override default of (11, 8)."""

    results = crossvalidate.validate_param_range(
                                X, y, model_type, param_name, param_range,
                                [score], other_params, splits, scale_obj)

    means = results.groupby(level=0).mean().reset_index()
    stds = results.groupby(level=0).std().reset_index()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(means.index)

    if semilog is True: plt_func = plt.semilogx
    else: plt_func = plt.plot
    plt_func(means.index, means[(score.__name__, "train")
                                ].values, 
             label="train", color="darkorange", lw=2)
    plt_func(means.index, means[(score.__name__, "test")
                                ].values, 
             label="validation", color="navy", lw=2)

    bands = lambda _: (means[(score.__name__, _)]
                        - stds[(score.__name__, _)],
                       means[(score.__name__, _)]
                        + stds[(score.__name__, _)])
    plt.fill_between(means.index, *bands("train"), 
                     alpha=0.1, color="darkorange", lw=2)
    plt.fill_between(means.index, *bands("test"), 
                     alpha=0.1, color="navy", lw=2)

    xlab = ax.set_xlabel(param_name)
    xlab = ax.set_xticklabels(means["index"].values)
    ylab = ax.set_ylabel(score.__name__)
    title = plt.title("validation curve: {}, across {}".format(
                      model_type.__name__, param_name))
    plt.legend(loc="best")


def calibration_curve(X, y, model_type, splits=5, model_params={},
                      calib_types=None, figsize=(11, 8), display_counts=True,
                      **cv_engine_kwargs):
    """
    Plots calibration curves for original model & passed calibrators.
    """
    def plot_probs_and_counts(results, c, label, plot_counts=display_counts):
        # greb emprirical probability from `results`
        probs = results["empirical_probability"]

        # plot empirical probability
        plt.plot(probs.index, probs["mean"], label=label, color=c, lw=2)

        # function to get +/- std around empirical probability means
        bands = lambda bin: (bin["mean"] - bin["std"], bin["mean"] + bin["std"])

        # plot counts with std lines if `display_counts` is True
        if plot_counts is True:
            counts = results["proportion_of_test_data"]
            plt.bar(counts.index, counts["mean"], yerr=counts["std"],
                    color="steelblue", width=.025, alpha=.2, label=None)

        # fill std around original model's mean calibration
        plt.fill_between(probs.index, *bands(probs), alpha=0.1, color=c, lw=2)

        # display horizontal grid lines
        plt.grid(which="major", axis="y", color='grey', linestyle='--')

    model_obj = model_type(**model_params)
    results = calibration.cv_calibrate(X=X, y=y,
                                       model_obj=model_obj,
                                       splits=splits,
                                       calib_types=calib_types,
                                       retain_counts=display_counts,
                                       **cv_engine_kwargs)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(results.index)
    colors = (c for c in ["C6", "C4", "C0"])

    plot = plot_probs_and_counts(results.loc[:, "original_model"], 
                                 colors.__next__(), 
                                 "original_model")

    if calib_types is not None:
        for cal_mod in calib_types:
            plot = plot_probs_and_counts(results.loc[:, cal_mod.__name__ + "_cal"], 
                                         colors.__next__(), 
                                         cal_mod.__name__,
                                         plot_counts=False)

    ax.set_xlim(0, 1)
    xlab = ax.set_xlabel("predicted probability (bin)")
    ax.set_ylim(0, 1)
    ylab = ax.set_ylabel("empirical probability")
    title = plt.title("calibration curve: {}".format(
                                model_type.__name__))
    plt.legend(loc="best")


def precision_recall_curve(X, y, model_type, scale_obj=None, 
                           splits=5, model_params={}, figsize=(11, 8),
                           thresholds=[.1*x for x in range(1, 10)]):
    """
    Plot precision-recall curve over decision boundaries:
    [0, 1] for binary classification. 

    Pass `model_type`, `model_params` and `figsze` in same fashion 
    as for learning_curve. `splits` and `scale_obj` are same as
    for all cv_score functions.
    """
    results = boundaries.cv_score(X, y, model_type(**model_params), 
                                 [recall_score, precision_score],
                                 thresholds,
                                 splits, False, scale_obj)
    to_plot = results.unstack()["mean"]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(to_plot["recall_score"], to_plot["precision_score"], lw=2)
    plt.fill_between(to_plot["recall_score"], to_plot["precision_score"],
                     alpha=.2)

    for row in to_plot.itertuples():
        ax.annotate("{}".format(round(row[0], 3)),
                    xy=(row[1], row[2]),
                    xytext=(row[1] - .01, 
                            row[2] + .02))
    
    plt.xlabel("pecall")
    plt.ylabel("precision")
    title = plt.title("precision & recall by decision boundary: {}".format(
                                        model_type.__name__))