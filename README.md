# modeling

#### Note

This repo (including its documentation) is still under construction, but currently available functionality is detailed briefly below.

#### Current functionality 

This *Python* repo provides some data analysis, class balancing, and cross validation tools to be used with *Pandas* data and *scikit-learn* models. Built using *Python* 3.6, *Pandas* .2, *sklearn* .18 and *Numpy* 1.12

**exploratory.center_scale_plot** produces histogram of provided *Series* w/ center & scale viz. & annotations

- user supplies center and scale functions

**evaluation.cv_score** k-fold cross validation for regression or binary classification 

- based on user-provided score function(s)

- score function object(s) must be passed in iterable

- can handle multiclass if passed lambda or decorator for each score function object

- where wrapper returns score function instantiated with applicable *average* kwarg value

- returns descriptive statistics for each score

- will return both train and test results unless user passes *False* for kwarg *train_scores* 

**balance.balance_binary_classes** takes *Series* of responses and returns roughly class-balanced subsamples
- for binary classes

- returns list of *n* index arrays, where *n* is rounded ratio of major class instances to minor class instances

**balance.cv_score** performs k-fold cv across user-provided balanced index arrays, similar to *evaluation.cv_score*

- the complete set of scores for each trial, for each index array, is used for calculating returned values

Additionally, the repo contains some lower level functions that produce, for example, unaggregated by-trial by-index scores.   
