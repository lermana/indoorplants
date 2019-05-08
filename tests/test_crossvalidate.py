import unittest
import collections
import numpy as np
import pandas as pd

from indoorplants.validation import crossvalidate


class ModelStubBase:

    def __init__(self, **kwargs):
        self.X_fit_shape = None
        self.y_fit_shape = None
        self.fit_called = False
        self.fit_called_num = 0

        self.X_predict_shape = None
        self.predict_called = False
        self.predict_called_num = 0

        for k, v in kwargs.items():
            self.k = v

    def fit(self, X, y):
        self.X_fit_shape = X.shape
        self.y_fit_shape = y.shape
        self.fit_called = True
        self.fit_called_num += 1
        return self

    def predict(self, X):
        self.X_predict_shape = X.shape
        self.predict_called = True
        self.predict_called_num += 1
        return np.zeros(self.y_fit_shape)


class RegressorStub(ModelStubBase):

    _estimator_type = "regressor"


class ClassifierStub(ModelStubBase):

    _estimator_type = "classifier"

    def fit(self, X, y):
        self.X_fit_shape = X.shape
        self.y_fit_shape = y.shape

        if len(self.y_fit_shape) == 1:
            self.num_classes = y.nunique()
        else:
            self.num_classes = self.y_fit_shape[1]
        
        self.fit_called = True
        self.fit_called_num += 1
        return self

    def predict_proba(self, X):
        self.X_predict_proba_shape = X.shape
        self.predict_proba_called = True
        return np.zeros((X.shape[0], self.num_classes))


def get_dummy_x_y():
    X = pd.DataFrame(np.zeros((100, 10)))
    y = pd.Series(np.zeros(50)
                  ).append(pd.Series(np.ones(50)))
    return X, y


def dummy_score_func(y_true, y_pred):
    return np.mean(y_true) - np.mean(y_pred)


class TestCrossvalidate(unittest.TestCase):

    def test_train_and_score(self):

        # get dummy functionality and data

        score_funcs = [dummy_score_func, dummy_score_func]
        X_train, y_train = get_dummy_x_y()
        X_test, y_test = get_dummy_x_y()

        # with `train_scores=True`

        model_obj = ClassifierStub()

        results = crossvalidate.train_and_score(model_obj, score_funcs,
                                                X_train, y_train,
                                                X_test, y_test, 
                                                train_scores=True)

        self.assertEqual(len(score_funcs), len(results))
        self.assertTrue(all(map(lambda row: len(row) == 2, results)))

        self.assertTrue(model_obj.fit_called)
        self.assertEqual(model_obj.X_fit_shape[0], model_obj.y_fit_shape[0])
        self.assertEqual(model_obj.num_classes, 2)

        self.assertTrue(model_obj.predict_called)
        self.assertEqual(model_obj.X_fit_shape[1], model_obj.X_predict_shape[1])
        self.assertEqual(model_obj.num_classes, 2)

        # with `train_scores=False`

        model_obj = ClassifierStub()

        results = crossvalidate.train_and_score(model_obj, score_funcs,
                                                X_train, y_train,
                                                X_test, y_test, 
                                                train_scores=False)

        self.assertEqual(len(score_funcs), len(results))
        self.assertTrue(all(map(
                                lambda row: not isinstance(row, collections.Iterable), 
                                results
                            )))

        self.assertTrue(model_obj.fit_called)
        self.assertEqual(model_obj.X_fit_shape[0], model_obj.y_fit_shape[0])
        self.assertEqual(model_obj.num_classes, 2)

        self.assertTrue(model_obj.predict_called)
        self.assertEqual(model_obj.X_fit_shape[1], model_obj.X_predict_shape[1])
        self.assertEqual(model_obj.num_classes, 2)

    def test_cv_engine(self):

        # get dummy args
        X, y = get_dummy_x_y()
        score_funcs = [dummy_score_func, dummy_score_func, dummy_score_func]
        splits = 5

        # with `model_obj._estimator_type="classifier"` and `train_scores=True`
        model_obj = ClassifierStub()
        train_scores = True

        results = crossvalidate.cv_engine(X, y, model_obj, score_funcs,
                                          splits, train_scores=train_scores)

        self.assertEqual(len(results), splits)

        for score_func_tuple in results:
            self.assertEqual(len(score_func_tuple), len(score_funcs))

            for train_test_scores in score_func_tuple:
                self.assertEqual(len(train_test_scores), 2)

        self.assertEqual(model_obj.fit_called_num, splits)
        self.assertEqual(model_obj.predict_called_num, 2 * splits)

        # with `model_obj._estimator_type="classifier"` and `train_scores=False`
        model_obj = ClassifierStub()
        train_scores = False

        results = crossvalidate.cv_engine(X, y, model_obj, score_funcs,
                                          splits, train_scores=train_scores)

        self.assertEqual(len(results), splits)

        for score_func_tuple in results:
            self.assertEqual(len(score_func_tuple), len(score_funcs))

            for score in score_func_tuple:
                self.assertTrue(isinstance(score, float))

        self.assertEqual(model_obj.fit_called_num, splits)
        self.assertEqual(model_obj.predict_called_num, splits)

        # with `model_obj._estimator_type="regressor"` and `train_scores=True`
        model_obj = RegressorStub()
        train_scores = True

        results = crossvalidate.cv_engine(X, y, model_obj, score_funcs,
                                          splits, train_scores=train_scores)

        self.assertEqual(len(results), splits)

        for score_func_tuple in results:
            self.assertEqual(len(score_func_tuple), len(score_funcs))

            for train_test_scores in score_func_tuple:
                self.assertEqual(len(train_test_scores), 2)

        self.assertEqual(model_obj.fit_called_num, splits)
        self.assertEqual(model_obj.predict_called_num, 2 * splits)

        # with `model_obj._estimator_type="regressor"` and `train_scores=False`
        model_obj = RegressorStub()
        train_scores = False

        results = crossvalidate.cv_engine(X, y, model_obj, score_funcs,
                                          splits, train_scores=train_scores)

        self.assertEqual(len(results), splits)

        for score_func_tuple in results:
            self.assertEqual(len(score_func_tuple), len(score_funcs))

            for score in score_func_tuple:
                self.assertTrue(isinstance(score, float))

        self.assertEqual(model_obj.fit_called_num, splits)
        self.assertEqual(model_obj.predict_called_num, splits)