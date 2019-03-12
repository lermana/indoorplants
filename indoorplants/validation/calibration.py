import numpy as np
import pandas as pd
from itertools import chain

from indoorplants.validation import crossvalidate


def _cv_proba(X, y, model_obj, splits=5, **cv_engine_kwargs):
    """
    Return numpy array of "splits"-fold CV results, 
    where results are:
    -column 0: actual class
    -columns 1 & 2: proba. of each of neg. and pos. classes.
    """
    model_obj.predict = model_obj.predict_proba

    score = lambda true, predicted: np.hstack(
                [true.values.reshape(-1, 1), predicted[:, 1].reshape(-1, 1)]
                )

    return crossvalidate.cv_engine(
                            X=X, y=y,
                            model_obj=model_obj,
                            score_funcs=[score],
                            splits=splits,
                            **cv_engine_kwargs
                            )


def _get_rank_stats(results):
    """
    Helper function that produces median and m.a.d
    probabilities for each of 2 classes given _cv_proba 
    results with reworked columns.
    """
    grouped = results[["class", "proba"]
                     ].groupby(
                        ["class"]
                     ).median(
                     ).rename(columns={"proba":"median"})

    return grouped.join(results[
                            ["class", "proba"]
                                ].groupby(["class"]
                  ).mad(
                  ).rename(columns={"proba":"mad"}))


def cv_rank(X, y, model_obj, splits=5, **cv_engine_kwargs):
    """
    Returns median and m.a.d. probabilities for each class
    for test results over "splits"-fold CV.
    """
    results = pd.DataFrame(
                np.vstack(
                    chain(
                        *_cv_proba(
                            X=X, y=y,
                            model_obj=model_obj,
                            splits=splits,
                            **cv_engine_kwargs
                    ))),
                columns=["class", "proba"])

    return _get_rank_stats(results)


def _calibrate_cv(model_results, calib_type, splits):
    """
    Produces calibrated probabilities for passed "model_results"
    using passed "calib_type" calibration model type 
    (uninstantiated model object).
    """
    def calibrator_fit(split_number):
        X = model_results.loc[split_number, "proba"].to_frame()
        y = model_results.loc[split_number, "class"]
        return calib_type().fit(X, y)

    def calibrator_predict_proba(calib_obj, split_number):
        X = model_results.loc[split_number, "proba"].to_frame()
        return calib_obj.predict_proba(X)

    def get_results_for_split(split_number):
        calib_obj = calibrator_fit(split_number)
        return calibrator_predict_proba(calib_obj, split_number)

    calib_res = [np.hstack(
                    [
                        model_results.loc[i, "class"].values.reshape(-1, 1),
                        get_results_for_split(i)[:, 1].reshape(-1, 1)
                    ])
                 for i in range(splits)]

    df = pd.concat({
            i: pd.DataFrame(data) for i, data in enumerate(calib_res)
            })

    df.columns = ["class", "proba"]
    return df


def _cv_calibrate(X, y, model_obj, splits=5, calib_types=None, **cv_engine_kwargs):
    """
    Returns CV results for passed model, and calibrates results.
    """
    model_res = _cv_proba(
                    X=X, y=y,
                    model_obj=model_obj,
                    splits=splits,
                    **cv_engine_kwargs
                    )

    df = pd.concat({
            i: pd.DataFrame(data)for i, data in enumerate(chain(*model_res))
        })

    df.columns = ["class", "proba"]

    if calib_types is None:
        return pd.concat({"original_model": df}, axis=1)

    else:
        dfs_cal = pd.concat(
                    [
                        pd.concat({
                            cal_type.__name__ + "_cal": _calibrate_cv(df, cal_type, splits)
                            }, axis=1) 
                        for cal_type in calib_types
                    ])

        return pd.concat({"original_model": df}, axis=1).join(dfs_cal)


def _group_by_fold_class_bin(df):
    """
    Meant to receive output from `_cv_calibrate`.
    """
    return df[["class", "prob_bin"]
             ].groupby(
                [df.index.get_level_values(0),"class", "prob_bin"]
             ).size(
             ).to_frame(
             ).rename(
                columns={0: "count"})


def _get_fold_bin_count(grouped):
    """
    Meant to receive output from `_group_by_fold_class_bin`.
    """
    grouped = grouped.groupby(level=[0, 2]
                    ).sum(
                    ).reset_index(
                    ).rename(
                        columns={"level_0": "fold","count": "count_prob_bin"}
                        )

    grouped['proportion_of_bin'] = grouped['count'] / grouped['count_prob_bin']
    return grouped


def _custom_round_to_int(x, base=5):
    return int(base * round(float(x)/base))


def _custom_round_to_five_hundredth(x):
    return _custom_round_to_int(100 * x) / 100


def _prob_bin_stats(results, pos_only=True, retain_counts=True):
    """
    Bins passed probabilities and calculates mean and std. 
    actual positive class frequencies.
    """
    df = results.copy()
    df["prob_bin"] = df.proba.apply(_custom_round_to_five_hundredth)

    grouped = _group_by_fold_class_bin(df)
    grouped = grouped.reset_index()

    grouped = grouped.merge(
                        _get_fold_bin_count(grouped),
                        on=["fold", "prob_bin"]
                    )

    grouped = grouped.set_index(["fold", "class", "prob_bin"])

    final = grouped.unstack(level=1
                  ).loc[:, "proportion_of_bin"
                  ].fillna(0)

    if not retain_counts:
        final = final.drop("count", axis=1)

    final.columns = final.columns.swaplevel()

    # TODO:
        # make sure my changes here work
        # keep going with the cleanup - knock out the below concat shit
        # ensure count and proportions are returned neatly from this func
        # and then ensure that this all works with cv_calibrate
        # and deeennn get this shit in the plot
    final = pd.concat(
                {"mean": final.groupby(level=[1]).mean()},
                    axis=1).join(pd.concat(
                        {"std": final.groupby(level=[1]).std()},
                             axis=1)).swaplevel(0, 1, 1)

    if pos_only is True:
        return final.loc[:, 1.0]

    else:
        return final


def cv_calibrate(X, y, model_obj, splits=5, calib_types=None,
                pos_only=True, **cv_engine_kwargs):
    """Return mean and std. CV results comparing actual positive
    class probabilities to binned predicted probabilities, for 
    the original model and all passed calibrators."""
    res = _cv_calibrate(
                    X=X, y=y,
                    model_obj=model_obj,
                    splits=splits,
                    calib_types=calib_types,
                    **cv_engine_kwargs
                    )

    modeled = pd.concat({"original_model":
                _prob_bin_stats(res.loc[:, "original_model"])}, axis=1)

    if calib_types is None:
        return modeled

    else:
        calib_results = [pd.concat({mod.__name__ + "_cal": 
                            _prob_bin_stats(res.loc[:, mod.__name__ + "_cal"])},
                            axis=1) for mod in calib_types]
        return modeled.join(pd.concat(calib_results))