import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score

import sklearn.model_selection as skl
import matplotlib.pyplot as plt

from indoorplants.validation import crossvalidate, \
                                    calibration, \
                                    boundaries


def validation_curve(X, y, score, model_type, param_name,
                     param_range, other_params={},
                     semilog=False, figsize=(11, 8),
                     **cv_engine_kwargs):
    """
    Cross validates `model_type` across passed parameters and
    plots results. Please see _validate_param_range for more
    details around the cross validation arguments.

    Pass True for `semilog` if `param_range` values would be better
    visualized with log scaling. Pass tuple to `figsize` if you 
    wish to override default of (11, 8).

    Please see `validation.cv_engine` for details on other args.
    """
    results = crossvalidate.validate_param_range(
                                X, y, model_type, param_name, param_range,
                                [score], other_params, **cv_engine_kwargs)

    # special handling for iterable hyper-parameter values
    if results.index.levels[0].dtype == "O":

        # map `param_range` values to `str`
        param_range = list(map(str, param_range))

        # get `Categorical` column, which will preserve original ordering
        cat_type = pd.api.types.CategoricalDtype(param_range, ordered=True)
        cat_df = pd.DataFrame(param_range, columns=[param_name]
                  ).astype(cat_type)

        # temporarily drop score function names from `results`
        score_name = results.columns.levels[0][0]
        results.columns = results.columns.droplevel(0)

        # merge in `Categorical` columns
        results = results.reset_index(
                        ).rename(columns={param_name: "param_value_str"})

        results = results.merge(cat_df, left_on="param_value_str", right_on=param_name
                        ).drop("param_value_str", axis=1)

        # sort `results` correctly and set index
        results = results.sort_values(param_name
                        ).set_index([param_name, "level_1"])

        # set `score_name` as top level column
        results = pd.concat({score_name: results}, axis=1)

    # get stats
    means = results.groupby(level=0).mean().reset_index()
    stds = results.groupby(level=0).std().reset_index()

    # create and setup figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(means.index)

    # handle logarithmic axes
    if semilog is True:
        plt_func = plt.semilogx
    else:
        plt_func = plt.plot

    # plot average train scores
    plt_func(means.index, means[(score.__name__, "train")].values, 
             label="train", color="darkorange", lw=2)

    # plot average validation scores
    plt_func(means.index, means[(score.__name__, "test")].values, 
             label="validation", color="navy", lw=2)

    # plot standard deviation bands around average scores
    bands = lambda _: (means[(score.__name__, _)] - stds[(score.__name__, _)],
                       means[(score.__name__, _)] + stds[(score.__name__, _)])

    plt.fill_between(means.index, *bands("train"), alpha=0.1, color="darkorange", lw=2)
    plt.fill_between(means.index, *bands("test"), alpha=0.1, color="navy", lw=2)

    # final plot formatting
    xlab = ax.set_xlabel(param_name)
    xlab = ax.set_xticklabels(means[param_name].values)
    ylab = ax.set_ylabel(score.__name__)
    title = plt.title(f"validation curve: {model_type.__name__}, across {param_name}")
    plt.legend(loc="best")


def learning_curve(X, y, model_type, score, model_params=None,
                   splits=5, scale_obj=None, random_state=0,
                   train_sizes=[0.1, 0.33, 0.55, 0.78, 1.],
                   figsize=(11, 8)):
    """
    Cross validates 'model_type' over different `train_sizes`, which allows 
    for both insight into model fit and insight into whether model might 
    benefit from access to more data. Plots results.

    Please see `validation.cv_engine` for details on other args.
    """
    if model_params is None:
        model_params = {}
    model_obj = model_type(**model_params)

    results = crossvalidate.validate_train_sizes(
                                X, y, model_obj, score_funcs=[score], 
                                splits=splits, scale_obj=scale_obj,
                                train_sizes=train_sizes,
                                random_state=random_state)

    means = results.groupby(level=0).mean().reset_index()
    stds = results.groupby(level=0).std().reset_index()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    xtick = ax.set_xticks(means.index)

    plt.plot(means.index, means[(score.__name__, "train")
                                ].values, 
             label="train", color="crimson", lw=2)
    plt.plot(means.index, means[(score.__name__, "test")
                                ].values, 
             label="validation", color="teal", lw=2)

    bands = lambda _: (means[(score.__name__, _)]
                        - stds[(score.__name__, _)],
                       means[(score.__name__, _)]
                        + stds[(score.__name__, _)])

    plt.fill_between(means.index, *bands("train"), 
                     alpha=0.1, color="crimson", lw=2)
    plt.fill_between(means.index, *bands("test"), 
                     alpha=0.1, color="teal", lw=2)

    xlab = ax.set_xlabel("training size")
    xlab = means["index"].map(lambda _: int(round(float(_) * len(y), -2)))

    xlab = ax.set_xticklabels(xlab)
    ylab = ax.set_ylabel(score.__name__)

    title = plt.title(f"Learning curve: {model_type.__name__}")
    plt.legend(loc="best")


def calibration_curve(X, y, model_type, model_params={}, calib_types=None, 
                      figsize=(11, 8), display_counts=True, **cv_engine_kwargs):
    """
    Plots calibration curves for original model & passed calibrators.

    Please see `validation.cv_engine` for details on other args.
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
                    color="steelblue", width=.025, alpha=.2, label="bin_count_normalized")

        # fill std around original model"s mean calibration
        plt.fill_between(probs.index, *bands(probs), alpha=0.1, color=c, lw=2)

        # display horizontal grid lines
        plt.grid(which="major", axis="y", color="grey", linestyle="--")

    model_obj = model_type(**model_params)
    results = calibration.cv_calibrate(X=X, y=y,
                                       model_obj=model_obj,
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

    title = plt.title(f"calibration curve: {model_type.__name__}")
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
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    title = plt.title("precision & recall by decision boundary: {}".format(
                                        model_type.__name__))