import pandas
import numpy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix



def split_data(y, n_splits=5, random_state=0):
    """input: variable to be predicted, [n_splits, random_state].
    returns: shuffled StratifiedKFold - iterator of n 
    train_index, test_index pairs."""
    skf = StratifiedKFold(n_splits=n_splits, 
                          shuffle=True, 
                          random_state=random_state)
    return skf.split(numpy.zeros(len(y)),
                     y)


def train_and_test(model, X_train, y_train, X_test, y_test):
    """input: instantiated model, training & test data sets.
    returns: training accuracy, test accuracy."""
    model = model.fit(X_train, 
                      y_train)

    return model.score(X_train, 
                       y_train), model.score(X_test, 
                                                y_test)


def get_model_variants(model, params_list):
    """input: uninstantiated model object, list of model param dictionaries.
    returns: generator of instantiated (model(params), params) tuples."""
    return ((model(**params), params) for params in params_list)
    

def get_models(model, params_list):
    """wrapper for get_model_variants.
    same inputs but handles exceptions due to no params_list,
    in which case a list of (model(), 'Default') is returned."""
    try:
        return get_model_variants(model, params_list)
    except TypeError:
        return [(model(), 'Default')]


def five_fold_validation(model_func, X, y, params_list=None):
    """input: uninstantiated model object, X, y, [params_list].
    returns: DataFrame of test and train accuracies for each of
    5 shuffled stratified trials, for each set of 
    model parameters."""
    results = []
    for fold, (train_index, test_index) in enumerate(split_data(y), 1):
        for model, params in get_models(model_func, params_list):
            train_acc, \
            test_acc = train_and_test(model, 
                                      X.iloc[train_index], 
                                      y.iloc[train_index],
                                      X.iloc[test_index],
                                      y.iloc[test_index])
            try:
                params_str = ', '.join('{}: {}'.format(param, 
                									   val) 
                                        for param, val in params.items())
            except AttributeError:
                params_str = params
            results.append((fold,
            	            params_str,
            	            train_acc,
            	            test_acc))
    cols = ['trial',
    		'params',
    		'train_accuracy',
    		'test_accuracy']
    return pandas.DataFrame(results,
                             columns=cols).set_index(['trial', 
                                                      'params'],
                                                     drop=True)


def validate_multiple_models(model_funcs, X, y, params_list=None):
    """input: same as five_fold_validation, except model_funcs 
    must be an iterator. Note that all models will run off the same
    params_list.
    returns: concatenated results of five_fold_validation."""
    results = []
    for model_func in model_funcs:
        model_name = str(model_func).split()[1].split('.')[3][:-2]
        results.append(pandas.concat({model_name:
                                      five_fold_validation(model_func, 
                                                           X, 
                                                           y, 
                                                           params_list)},
                                      axis=1))
    return pandas.concat(results, axis=1)


def get_confusion(y_test, y_hat, index):
    """returns sklearn confusion matrix as DataFrame,
    with index serving as labels for rows and columns."""
    return pandas.DataFrame(confusion_matrix(y_test, 
                                             y_hat),
                            index=index,
                            columns=index)


def get_precision(y_test, y_hat, index):
    """returns by-value precision as DataFrame."""
    return pandas.DataFrame(
                numpy.diag(
                    confusion_matrix(y_test,
                                     y_hat)
                         / confusion_matrix(y_test,
                                            y_hat).sum(axis=0)
                 ), index=index,
                    columns=['Precision'])


def get_recall(y_test, y_hat, index):
    """returns by-value recall as DataFrame."""
    return pandas.DataFrame(
                numpy.diag(
                    confusion_matrix(y_test,
                                     y_hat)
                         / confusion_matrix(y_test,
                                            y_hat).sum(axis=1)
                 ), index=index,
                    columns=['Recall'])