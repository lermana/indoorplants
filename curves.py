import numpy as np
import sklearn.model_selection as skl
import matplotlib.pyplot as plt


def learning_curve(model_func, X, y, cv=5, score=None,
                   train_sizes=np.linspace(.1, 1.0, 5)):
    
    train_sizes, train_scores, test_scores = \
                skl.learning_curve(model_func(), 
                                   X, 
                                   y,
                                   cv=cv, 
                                   scoring=score,
                                   train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(11, 8))
    plt.xlabel("train size")
    plt.ylabel(score)
    title = plt.title('Learning Curve: {}'.format(
                            model_func.__name__))

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


def validation_curve(model_func, X, y, param_name, 
                     param_range, cv=5, score=None,
                     semilog=False):
    
    train_scores, test_scores = \
                skl.validation_curve(model_func(), 
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
        plt.semilogx(param_range, test_scores_mean, 
                     label="validation", color="navy", lw=2)
    plt.fill_between(param_range, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color="navy", lw=2)

    plt.legend(loc="best")