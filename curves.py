import numpy as np
import sklearn.model_selection as skl
import matplotlib.pyplot as plt



def _plot_learning(func):
    """wrapper to plot CV results as learning curve""" 
    def plot(*args, **kwargs):
        results = func(*args, **kwargs)

        train_sizes = results['train_sizes']
        train_scores = results['train_scores']
        test_scores = results['test_scores']
        model_name = results['model_name']
        score = results['score']

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(11, 8))
        plt.xlabel("train size")
        plt.ylabel(score)
        title = plt.title('Learning Curve: {}'.format(
                                        model_name))

        plt.plot(train_sizes, train_scores_mean, 
                 color="crimson", label="train", lw=2)

        plt.fill_between(train_sizes, 
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, 
                         alpha=0.1, color="crimson", lw=2)

        plt.plot(train_sizes, test_scores_mean, 
                 color="teal", label="validation", lw=2)

        plt.fill_between(train_sizes, 
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, 
                         alpha=0.1, color="teal", lw=2)

        plt.legend(loc="best")
    return plot


@_plot_learning
def learning_curve(model_func, model_params=None, score=None, cv=5,
                   X=None, y=None, bal_inds=None):
    """*still debugging*: function may have issues.

    helper func that wraps sklearn's learning curve function.
    allows for multiple rounds of CV (each done cv times) due to 
    class-balanced resampling or the typlical cv-fold CV."""
    
    get_results = lambda X, y: skl.learning_curve(
        model, X, y,cv=cv, scoring=score)
    results = {}

    if model_params is not None: model = model_func(**model_params)
    else: model = model_func()
    
    if bal_inds is not None:
        X_, y_ = X[X.index.isin(bal_inds[0])], y[y.index.isin(bal_inds[0])]
        train_sizes, train_scores, test_scores = get_results(X_, y_)
        i, num = 1, len(bal_inds)
        while i < num:
            X_, y_ = X[X.index.isin(bal_inds[i])], y[y.index.isin(bal_inds[i])]
            _, train_, test_ = get_results(X_, y_)
            train_scores = np.append(train_scores, train_, axis=1)
            test_scores = np.append(test_scores, test_, axis=1)
            i += 1
    else:
        train_sizes, train_scores, test_scores = get_results(X, y)

    results['train_sizes'] = train_sizes
    results['train_scores'] = train_scores
    results['test_scores'] = test_scores
    results['model_name'] = model_func.__name__
    results['score'] = score
    return results
    


def validation_curve(model_func, X, y, param_name, 
                     param_range, cv=5, score=None,
                     semilog=False, other_params=None):
    """wraps sklearn's validation curve"""
    if other_params is not None:
        model = model_func(**other_params)
    else:
        model = model_func()

    train_scores, test_scores = \
                skl.validation_curve(model, 
                                     X, 
                                     y,
                                     param_name=param_name, 
                                     param_range=param_range,
                                     cv=cv, 
                                     scoring=score)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(11, 8))
    title = plt.title('Validation Curve: {}, across {}'.format(
                      model_func.__name__,
                      param_name))

    xlab = plt.xlabel(param_name)
    ylab = plt.ylabel(score)
    plt.ylim(0.0, 1.1)
    
    if semilog is True:
        plt.semilogx(param_range, train_scores_mean, 
                     label="train", color="darkorange", lw=2)
    else:
        plt.plot(param_range, train_scores_mean, 
                 label="train", color="darkorange", lw=2)
    plt.fill_between(param_range, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color="darkorange", lw=2)
    
    if semilog is True:
        plt.semilogx(param_range, test_scores_mean, 
                     label="validation", color="navy", lw=2)
    else:
        plt.plot(param_range, test_scores_mean, 
                 label="validation", color="navy", lw=2)
    plt.fill_between(param_range, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color="navy", lw=2)

    plt.legend(loc="best")