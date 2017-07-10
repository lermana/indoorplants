import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, \
                            accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score, \
                            mean_squared_error, \
                            r2_score


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


def _cv(X, y, model, score_func, splits=5, scale_obj=None, 
        param_name=None, param_range=None, train_sizes=None,
        train_scores=True):
        
        results = []
        skf = StratifiedKFold(n_splits=splits,
                              random_state=0)

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


def precision_recall(y, y_hat):
    return precision_score(y, y_hat), recall_score(y, y_hat)


def cv_prec_rec(X, y, model, splits=5, scale_obj=None):
    
    results = _cv(X, y, model, precision_recall, 
                  splits, scale_obj)
    results = np.array([tuple(chain(*trial)) for trial in results])
    
    train = pd.concat({'train': pd.DataFrame(results[:, :2], 
                                    columns=['precision',
                                              'recall'])},
                      axis=1)
    test = pd.concat({'test': pd.DataFrame(results[:, :2], 
                                    columns=['precision',
                                             'recall'])},
                      axis=1)
    
    return train.join(test).mean().rename('score').to_frame()


def cv_conf_mat(X, y, model, splits=5, scale_obj=None):
    results = _cv(X, y, model, confusion_matrix, 
                  splits, scale_obj, train_scores=False)
    
    results = [pd.concat({i: pd.DataFrame(trial,
                                index=['neg_true', 'pos_true'],
                                columns=['neg_pred', 'pos_pred'])}) 
                   for i, trial in enumerate(results, 1)]
    
    return pd.concat(results)


def get_model_name(model_func):
    return str(model_func).split('(')[0]


def plot_validation_curve(model_func, X, y, param_name, 
                          param_range, cv=5, scoring='f1',
                          semilog=False):
    
    train_scores, test_scores = \
                validation_curve(model_func, 
                                 X, 
                                 y,
                                 param_name=param_name, 
                                 param_range=param_range,
                                 cv=cv, 
                                 scoring=scoring)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(11, 8))
    title = plt.title('Validation Curve: {}, across {}'.format(
                            get_model_name(model_func),
                            param_name))

    xlab = plt.xlabel(param_name)
    ylab = plt.ylabel(scoring)
    plt.ylim(0.0, 1.1)
    
    if semilog is True:
        plt.semilogx(param_range, train_scores_mean, 
                     label="Training score", color="darkorange", lw=2)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=2)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=2)
    if semilog is True:
        plt.semilogx(param_range, test_scores_mean, 
                     label="Cross-validation score", color="navy", lw=2)
    else:
        plt.semilogx(param_range, test_scores_mean, 
                     label="Cross-validation score", color="navy", lw=2)
    
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=2)

    plt.legend(loc="best")