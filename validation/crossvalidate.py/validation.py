import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix


def _train_and_score(model_obj, score_funcs, X_train, y_train,
                     X_test, y_test, train_scores=True):
    """Trains model to training data, outputs score(test data) 
    for each score in 'score_funcs', and does the same for 
    train data unless 'train_data' is set to False"""
    model = model_obj.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    return [(score_func(y_train, 
                        y_hat_train),
            score_func(y_test, y_hat_test))
            if train_scores is True else
            (score_func(y_test, y_hat_test))
            for score_func in score_funcs]


def _cv_engine(X, y, model_obj, score_funcs, splits=5,
               scale_obj=None, train_scores=True):
    """Splits data (based on whether model is classifier
    or regressor) and passes each fold to the train_and_score
    function. 

    Collects results and returns results as List."""
    if model_obj._estimator_type == 'classifier':
        skf = StratifiedKFold(n_splits=splits,
                              random_state=0)
    elif model_obj._estimator_type == 'regressor':
        skf = KFold(n_splits=splits,
                    suffle=True,
                    random_state=0)
    else:
        raise TypeError('Improper model type.')

    results = []
    for train, test in skf.split(X, y):
        
        if scale_obj is not None:
            X_train = scale_obj.fit_transform(X.iloc[train, :])
            X_test = scale_obj.fit_transform(X.iloc[test, :])
        else:
            X_train = X.iloc[train, :]
            X_test = X.iloc[test, :]
            
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        
        results.append(_train_and_score(model_obj, score_funcs, 
                                        X_train, y_train,
                                        X_test, y_test, 
                                        train_scores))
    return results


def _cv_format(X, y, model_obj, score_funcs, splits=5,
               scale_obj=None, train_scores=True):
    """Gets results from _cv_engine and returns as 
    unaggregated DataFrame, where trial number & score 
    function used are represented in index."""
    res = _cv_engine(X, y, model_obj, score_funcs, 
                     splits, scale_obj, train_scores)
    if train_scores is False:
        cols = [_.__name__ for _ in score_funcs]
        return pd.DataFrame(res, columns=cols)
    else:
        res = np.array([tuple(chain(*_)) for _ in res])
        i, n, dfs = 0, res.shape[1], []
        while i < n:
            dfs.append(pd.concat({
                score_funcs[i // 2].__name__:
                    pd.DataFrame(res[:, i:i+2], 
                                columns=['train', 'test'])},
                    axis=1))
            i += 2
        return dfs[0].join(dfs[1:])


def _cv_score(results):
    """Given DF of CV results, returns DF of descriptive 
    statistics. Which stats are returned is not currently
    controlled by user but should be."""
    return results.mean().rename('mean').to_frame(
            ).join(results.std().rename('std').to_frame()
            ).join(results.skew().rename('skew').to_frame()
            ).join(results.kurtosis().rename('kurt').to_frame()
            ).join(results.min().rename('min').to_frame()
            ).join(results.max().rename('max').to_frame())


def cv_score(X, y, model_obj, score_funcs, splits=5,
             scale_obj=None, train_scores=True):
    """Returns DataFrame of stats on 'model_obj' 
    cross-validated over 'splits' splits of data,
    using scores in 'score_funcs', which must be 
    an interable. 

    If 'scale_obj' is passed, X will be scaled within
    each fold so as to prevent data leakage.

    'train_scores' [default True] specifies reporting 
    on train scores."""
    return _cv_score(_cv_format(X, y, model_obj, 
                                score_funcs, splits, 
                                scale_obj, train_scores))


def cv_conf_mat(X, y, model_obj, splits=5, scale_obj=None):
    """Return confusion matrix for each CV trial."""
    results = _cv_engine(X=X, y=y, model_obj=model_obj, 
                        score_funcs=[confusion_matrix], 
                        splits=splits, scale_obj=scale_obj, 
                        train_scores=False)
    
    results = [pd.concat({i: pd.DataFrame(trial[0],
                            index=['neg_true', 'pos_true'],
                            columns=['neg_pred', 'pos_pred'])}) 
                   for i, trial in enumerate(results, 1)]
    
    return pd.concat(results)


