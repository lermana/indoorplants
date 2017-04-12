import pandas
import numpy

from sklearn.model_selection import StratifiedKFold, \
                                    KFold

from sklearn.metrics import confusion_matrix, \
                            accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score



def get_function(func_clf, func_reg, problem):
    try:
        return {'clf': func_clf, 'reg': func_reg}[problem]
    except KeyError:
        raise KeyError("Must pass either 'clf' or 'reg'.")


def generate_folds(fold_obj, y, n_splits=5, random_state=0):
    return fold_obj(n_splits=n_splits, 
                    shuffle=True, 
                    random_state=random_state).split(numpy.zeros(len(y)),
                                                     y)


def split_for_clf(y):
    return generate_folds(StratifiedKFold, y)


def split_for_reg(y):
    return generate_folds(KFold, y)


def get_split_func(problem):
    return get_function(split_for_clf, split_for_reg, problem)


def split_data(y, problem):
    return get_split_func(problem)(y)


def score_reg(model, X_train, y_train, X_test, y_test):
    return (model.score(X_train, y_train), 
            model.score(X_test, y_test))


def score_clf(model, X_train, y_train, X_test, y_test):
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    return (accuracy_score(y_train, 
                           y_hat_train),
            accuracy_score(y_test, 
                           y_hat_test),  
            precision_score(y_test, 
                            y_hat_test),
            recall_score(y_test, 
                         y_hat_test),
            f1_score(y_test, 
                     y_hat_test))


def get_score_func(problem):
    return get_function(score_clf, score_reg, problem)


def score_model(model, X_train, y_train, X_test, y_test, problem):
    return get_score_func(problem)(model, X_train, y_train, X_test, y_test)


def train_and_test(model, X_train, y_train, X_test, y_test, problem):
    model = model.fit(X_train, y_train)
    return score_model(model, 
                       X_train, 
                       y_train, 
                       X_test, 
                       y_test, 
                       problem)


def get_model_variants(model, params_list):
    return ((model(**params), params) for params in params_list)
    

def get_models(model, params_list):
    try:
        return get_model_variants(model, params_list)
    except TypeError:
        return ((model(), 'Default'),)


def results_dataframe(results, cols):
    return pandas.DataFrame(results,
                            columns=cols).set_index(['trial', 
                                                     'params'],
                                                    drop=True)


def format_clf_validation(results):
    cols = ['trial',
            'params',
            'train_accuracy',
            'test_accuracy',
            'test_precision',
            'test_recall',
            'test_f1score']
    return results_dataframe(results, cols)


def format_reg_validation(results):
    cols = ['trial',
            'params',
            'train_rsquared',
            'test_rsquared']
    return results_dataframe(results, cols)


def get_format(problem):
    return get_function(format_clf_validation, format_reg_validation, problem)

def format_validation(results, problem):
    return get_format(problem)(results)


def five_fold_validation(model_func, X, y, problem, params_list=None):
    results = []
    for fold, (train_index, test_index) in enumerate(split_data(y, problem), 1):
        for model, params in get_models(model_func, params_list):
            itersults = []
            itersults.append(fold)
            try:
                itersults.append(', '.join('{}: {}'.format(param, 
                                                           val) 
                                        for param, val in params.items()))
            except AttributeError:
                itersults.append(params)
            for score in train_and_test(model, 
                                        X.iloc[train_index], 
                                        y.iloc[train_index], 
                                        X.iloc[test_index], 
                                        y.iloc[test_index], 
                                        problem):
                itersults.append(score)
            results.append(itersults)
    return format_validation(results, problem)


def validate_multiple_models(model_funcs, X, y, params_list=None):
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
    return pandas.DataFrame(confusion_matrix(y_test, 
                                             y_hat),
                            index=index,
                            columns=index)


def get_precision(y_test, y_hat, index):
    return pandas.DataFrame(
                numpy.diag(
                    confusion_matrix(y_test,
                                     y_hat)
                         / confusion_matrix(y_test,
                                            y_hat).sum(axis=0)
                 ), index=index,
                    columns=['Precision'])


def get_recall(y_test, y_hat, index):
    return pandas.DataFrame(
                numpy.diag(
                    confusion_matrix(y_test,
                                     y_hat)
                         / confusion_matrix(y_test,
                                            y_hat).sum(axis=1)
                 ), index=index,
                    columns=['Recall'])