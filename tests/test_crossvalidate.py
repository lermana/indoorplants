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

        self.X_predict_shape = None
        self.predict_called = False

        for k, v in kwargs.items():
            self.k = v

    def fit(self, X, y):
        self.X_fit_shape = X.shape
        self.y_fit_shape = y.shape
        self.fit_called = True
        return self

    def predict(self, X):
        self.X_predict_shape = X.shape
        self.predict_called = True
        return np.zeros(self.y_fit_shape)


class ClassifierStub(ModelStubBase):

    def fit(self, X, y):
        self.X_fit_shape = X.shape
        self.y_fit_shape = y.shape

        if len(self.y_fit_shape) == 1:
            self.num_classes = y.nunique()
        else:
            self.num_classes = self.y_fit_shape[1]
        
        self.fit_called = True
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
        self.assertTrue(all(map(lambda row: not isinstance(row, collections.Iterable), results)))

        self.assertTrue(model_obj.fit_called)
        self.assertEqual(model_obj.X_fit_shape[0], model_obj.y_fit_shape[0])
        self.assertEqual(model_obj.num_classes, 2)

        self.assertTrue(model_obj.predict_called)
        self.assertEqual(model_obj.X_fit_shape[1], model_obj.X_predict_shape[1])
        self.assertEqual(model_obj.num_classes, 2)

    def test_cv_engine(self):

        # get dummy functionality and data

        score_funcs = [dummy_score_func, dummy_score_func]
        X_train, y_train = get_dummy_x_y()
        X_test, y_test = get_dummy_x_y()

        # test with and without train scores
        # test a couple split numbers
        # test with and without scale obj (ugh, you'll have to create a sub for this, too...)
        # test with one score func, two score funcs, three score funcs?
        # test with classification and regression - should you have different y for regression?