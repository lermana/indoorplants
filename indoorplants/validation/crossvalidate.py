import collections
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, \
                                    StratifiedShuffleSplit, learning_curve
from sklearn.metrics import confusion_matrix, make_scorer


def train_and_score(model_obj, score_funcs, X_train, y_train,
                     X_test, y_test, train_scores=True):
    """
    Trains model to training data, outputs score(test data) 
    for each score in "score_funcs", and does the same for 
    train data unless "train_data" is set to False
    """
    def apply_score_func(func):
        if train_scores is True:
            return (func(y_train, y_hat_train), func(y_test, y_hat_test))
        else:
            return func(y_test, y_hat_test)

    model = model_obj.fit(X_train, y_train)

    if train_scores is True:
        y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    return list(map(apply_score_func, score_funcs))


def cv_engine(X, y, model_obj, score_funcs, splits=5, scale_obj=None, 
              train_scores=True, random_state=0, cross_validator_obj=None):
    """
    Splits data (based on whether model is classifier
    or regressor) and passes each fold to the `train_and_score`
    function.

    Collects results and returns results as `list`.

    parameters
    ----------

    X: pd.DataFrame
        Exogenous variables to be used as model inputs.

    y: pd.Series
        Endogenous variable that should be predicted by the model.

    model_obj: sklearn.BaseEstimator
        Instantiated (i.e. this is an instance of a class to which 
        hyper-parameters have already been passed) scikit-learn model, or model 
        with similar API - i.e.`model_obj.fit(X, y)` and `model_obj.predict(X)` 
        would be used for model fitting and predicting, respectively. E.g. 
        `sklearn.RandomForestClassifier(max_depth=16)`.

    score_funcs: list-like of callables
        Score functions to be run on model predictions. E.g.
        `[sklearn.metrics.f1_score, sklearn.metrics.accuracy_score]`.

    splits: int, optional(default=5)
        Number of splits to use for k-fold cross-validation.

    scale_obj: sklearn.TransformerMixin (default=None)
        Should be a scikit-learn transform object (i.e. inherits from 
        `sklearn.BaseEstimator` and `sklearn.TransformerMixin`) or have 
        a similar API (i.e. `scale_obj.fit_transform()` works as expected). 
        If passed, X will be scaled within each fold so as to prevent data 
        leakage. 

    train_scores: bool (default=True)
        Determines whether training scores, in addition to test scores, are 
        returned. Train scores are useful for comparing to test scores in order 
        to assess model fit.

    `random_state`: int, RandomState instance or None (default=0)
        Set the seed used for random sampling in cross-validation, which allows 
        for reproducability of data splits. See sklearn docs, e.g. for 
        `StratifiedKFold`, for more information.

    `cross_validator_obj`: `sklean.model_selection._BaseKFold`
        Can pass a cross-validator to override defaul CV behavior (e.g. when
        geting results for a learning curve).

    return
    ------

    `list` of `train_and_score` results, with length `splits`.

    """
    if cross_validator_obj is not None:
        skf = cross_validator_obj

    elif model_obj._estimator_type == "classifier":
        skf = StratifiedKFold(n_splits=splits, shuffle=True,
                              random_state=random_state)

    elif model_obj._estimator_type == "regressor":
        skf = KFold(n_splits=splits, shuffle=True, random_state=random_state)

    else:
        raise TypeError("Improper model type.")

    results = []
    for train, test in skf.split(X, y):
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        X_train = X.iloc[train, :]
        X_test = X.iloc[test, :]

        if scale_obj is not None:
            X_train = scale_obj.fit_transform(X_train)
            X_test = scale_obj.fit_transform(X_test)

        results.append(
            train_and_score(
                    model_obj, score_funcs, 
                    X_train, y_train,
                    X_test, y_test,
                    train_scores
                ))

    return results


def format_cv_results(results, score_funcs, train_scores=True):
    """
    Takes results from cv_engine and returns as 
    unaggregated DataFrame, where trial number & score 
    function used are represented in index.
    """
    if train_scores is False:
        cols = [score.__name__ for score in score_funcs]
        return pd.DataFrame(results, columns=cols)

    else:
        res = np.array([tuple(chain(*trial)) for trial in results])
        dfs = []

        for i in range(0, res.shape[1], 2):
            dfs.append(
                pd.concat(
                    {
                    score_funcs[i // 2].__name__: 
                        pd.DataFrame(res[:, i:i+2], columns=["train", "test"])
                        },
                    axis=1
                    ))

        return dfs[0].join(dfs[1:])


def describe_dataframe(results, stats_to_run=["mean", "std"]):
    """
    Given DF of CV results, returns DF of descriptive statistics.

    parameters
    ----------

    results: pandas.DataFrame
        CV results, upon upon which descriptive statistics are to be run.

    stats_to_run: str or list-like of str
        pandas.DataFrame method name(s) indicating statistic to be run,
        e.g. "mean" or ["mean", "std"]

    return
    ------

    pandas.DataFrame of descriptive statistics, where rows correspond to
    columns of `results` and where each columns correspond to `stats_to_run`.

    """
    if isinstance(stats_to_run, str):
        stats_to_run = [stats_to_run]

    get_stats = lambda func_name: getattr(results, func_name)(
                                         ).rename(func_name
                                         ).to_frame()

    to_return = [get_stats(s) for s in stats_to_run]

    return to_return[0].join(to_return[1:])


def cv_score(X, y, model_obj, score_funcs, stats_to_run=["mean", "std"],
            train_scores=True, **cv_engine_kwargs):
    """
    Cross-validates passed model and returns performance statistics. 
    Cross-validiation is performed using shuffling. If passed model is a 
    classifier, shuffling is performed such that existing stratification 
    of classes is preserved.

    parameters
    ----------

    X: pd.DataFrame
        Exogenous variables to be used as model inputs.

    y: pd.Series
        Endogenous variable that should be predicted by the model.

    model_obj: sklearn.BaseEstimator
        Instantiated (i.e. this is an instance of a class to which 
        hyper-parameters have already been passed) scikit-learn model, 
        or model with similar API - i.e. `model_obj.fit(X, y)` and 
        `model_obj.predict(X)` would be used for model fitting and predicting, 
        respectively. E.g. `sklearn.RandomForestClassifier(max_depth=16)`.

    score_funcs: callable, or list-like of callables
        Score function(s) to be run on model predictions. E.g. 
        `sklearn.metrics.accuracy_score` or 
        `[sklearn.metrics.f1_score, sklearn.metrics.accuracy_score]`.

    stats_to_run: str or list-like of str, optional(default=["mean", "std"])
        pandas.DataFrame method name(s) indicating statistic to be run,
        e.g. "mean" or `["mad", "var"]`.

    train_scores: bool (default=True)
        Determines whether training scores, in addition to test scores, are 
        returned. Train scores are useful for comparing to test scores in order 
        to assess model fit.

    See `cv_engine` for more information on additional kwargs.

    return
    ------

    pandas.DataFrame of descriptive statistics, as specified in 
    `describe_dataframe`.
    """
    if callable(score_funcs):
        score_funcs = [score_funcs]

    return describe_dataframe(
                format_cv_results(
                    cv_engine(
                        X, y, model_obj, score_funcs, train_scores=train_scores,
                        **cv_engine_kwargs
                        ),
                    score_funcs, train_scores))


def cv_conf_mat(X, y, model_obj, **cv_engine_kwargs):
    """
    Return confusion matrix for each CV trial.

    See `cv_engine` for more information on additional kwargs.
    """
    results = cv_engine(X=X, y=y, model_obj=model_obj, 
                        score_funcs=[confusion_matrix],
                        train_scores=False,
                        **cv_engine_kwargs)

    results = [pd.concat(
                {
                    i: pd.DataFrame(trial[0],
                                    index=["neg_true", "pos_true"],
                                    columns=["neg_pred", "pos_pred"])
                }
               ) for i, trial in enumerate(results, 1)]

    return pd.concat(results)


def validate_param_range(X, y, model_type, param_name, param_range,
                         score_funcs, other_params={}, train_scores=True,
                         **cv_engine_kwargs):
    """
    Returns `validation.cv_score` across values in `param_range`
    for `param_name`, which should be a working parameter for the
    passed model.

    `model_type` should be an uninstantiated sklearn model (or
    one with similar fit and predict methods). Additional 
    hyper-parameters (i.e. not `param_name` should be passed
    in to `other_params` as dictionary.

    Please see `validation.cv_engine` for details on other args.
    """ 
    results = {}
    for val in param_range:
        model_obj = model_type(**{param_name: val}, **other_params)

        some_kwargs = {"model_obj": model_obj, "X": X, "y": y, 
                       **cv_engine_kwargs}

        other_kwargs = {"train_scores": train_scores,
                        "score_funcs": score_funcs}

        if isinstance(val, collections.Iterable):
            val = str(val)

        res = cv_engine(**some_kwargs, **other_kwargs)
        results[val] = format_cv_results(res, **other_kwargs)

    to_return = pd.concat(results)
    to_return.index = to_return.index.rename(param_name, level=0)

    return to_return


def validate_train_sizes(X, y, model_obj, score_funcs, splits=5,
                         train_sizes=[0.1, 0.33, 0.55, 0.78, 1.],
                         scale_obj=None, random_state=0):
    """
    Cross validates 'model_obj' over different `train_sizes`, which allows 
    for both insight into model fit and insight into whether model might 
    benefit from access to more data.

    Please see `validation.cv_engine` for details on other args.
    """
    if callable(score_funcs):
        score_funcs = [score_funcs]

    results = {}
    for size in train_sizes:
        train_size = (1 - (1 / splits)) * size

        cross_validator_obj = StratifiedShuffleSplit(n_splits=splits,
                                                     train_size=train_size,
                                                     test_size=1 / splits,
                                                     random_state=random_state)

        some_kwargs = {"model_obj": model_obj, "X": X, "y": y, 
                       "splits": splits, "scale_obj": scale_obj,
                       "cross_validator_obj": cross_validator_obj,
                       "random_state": random_state}

        other_kwargs = {"train_scores": True, "score_funcs": score_funcs}

        res = cv_engine(**some_kwargs, **other_kwargs)
        results[size] = format_cv_results(res, **other_kwargs)

    to_return = pd.concat(results)
    return to_return