import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix


def _train_and_score(model, X_train, y_train, X_test, y_test, 
                     score_func, train_scores):
    
    model = model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    if train_scores is True:
        return ((score_func(y_train, y_hat_train),
                 score_func(y_test, y_hat_test)))
    else:
        return (score_func(y_test, y_hat_test))


def _cv_engine(X, y, model, score_func, splits=5, scale_obj=None, 
        param_name=None, param_range=None, train_sizes=None,
        train_scores=True):
        
        if model._estimator_type == 'classifier':
            skf = StratifiedKFold(n_splits=splits,
                                  random_state=0)
        elif model._estimator_type == 'regressor':
            skf = KFold(n_splits=splits,
                        suffle=True,
                        random_state=0)
        else:
            raise TypeError('Improper model type.')

        results = []
        for train, test in skf.split(X, y):
            
            if scale_obj is not None:
                X_train = scale_obj().fit_transform(X.iloc[train, :])
                X_test = scale_obj().fit_transform(X.iloc[test, :])
            else:
                X_train = X.iloc[train, :]
                X_test = X.iloc[test, :]
                
            y_train = y.iloc[train]
            y_test = y.iloc[test]
            
            results.append(_train_and_score(model, X_train, y_train, 
                                            X_test, y_test, score_func, 
                                            train_scores))
        return results


def _get_score_name(score_func):
    return str(score_func).split()[1]


def _cv(X, y, model, score_func, splits=5, scale_obj=None):
    results = _cv_engine(X, y, model, score_func, splits, scale_obj)
    results = pd.DataFrame(results, columns=['train', 'test'])
    return pd.concat({_get_score_name(score_func): results})


def cv_score(X, y, model, score_func, splits=5, scale_obj=None):
    results = _cv(X, y, model, score_func, splits, scale_obj)
    return results.mean().rename('mean').to_frame(
            ).join(results.std().rename('std').to_frame())


def _get_two_scores(score_1, score_2):
    def scores(y, y_hat):
        return score_1(y, y_hat), score_2(y, y_hat)    
    return scores


def _cv_two(X, y, model, score_1, score_2, splits=5, scale_obj=None):
    scores = _get_two_scores(score_1, score_2)
    results = _cv_engine(X, y, model, scores, splits, scale_obj)
    
    results = np.array([tuple(chain(*trial)) for trial in results])
    s1, s2 = _get_score_name(score_1), _get_score_name(score_2)
    train = pd.concat({'train': pd.DataFrame(results[:, :2], 
                                    columns=[s1, s2])}, axis=1)
    test = pd.concat({'test': pd.DataFrame(results[:, :2], 
                                    columns=[s1, s2])},axis=1)
    return train.join(test)


def cv_two_scores(X, y, model, score_1, score_2, splits=5, scale_obj=None):
    results = _cv_two(X, y, model, score_1, score_2, splits, scale_obj)
    return results.mean().rename('mean').to_frame(
            ).join(results.std().rename('std').to_frame())


def cv_conf_mat(X, y, model, splits=5, scale_obj=None):
    results = _cv(X, y, model, confusion_matrix, 
                  splits, scale_obj, train_scores=False)
    
    results = [pd.concat({i: pd.DataFrame(trial,
                                index=['neg_true', 'pos_true'],
                                columns=['neg_pred', 'pos_pred'])}) 
                   for i, trial in enumerate(results, 1)]
    
    return pd.concat(results)


