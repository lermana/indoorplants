import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def euclidean(feature_one, feature_two):
    return np.sqrt(
                (feature_one - feature_two
                   ).pow(2
                   ).sum())


def euclidean_squared(feature_one, feature_two):
    return (feature_one - feature_two
               ).pow(2
               ).sum()


def get_feature_iter_func(flat):
    if flat:
        return lambda feature_names: itertools.combinations(feature_names, 2)
    else:
        return lambda feature_names: itertools.product(feature_names, feature_names)


def normalize_min_max(df):
    cols = df.columns
    inds = df.index

    arr = MinMaxScaler().fit_transform(df)
    return pd.DataFrame(arr, columns=cols, index=inds)


def feature_distances(distance_func):

    def inner(df, feature_names_list, flat=True, normalize_data=True):
        if normalize_data:
            df = normalize_min_max(df[feature_names_list].astype(float))

        iter_func = get_feature_iter_func(flat)

        scores = [(f1, f2, distance_func(df[f1], df[f2]))
                  for f1, f2 in iter_func(feature_names_list)]

        scores_df = pd.DataFrame(scores, columns=["feature_one",
                                                  "feature_two",
                                                  f"{distance_func.__name__}"])

        if flat:
            scores_df = scores_df.set_index(["feature_one",
                                             "feature_two"]
                                ).sort_values(f"{distance_func.__name__}", 
                                              ascending=False)

        else:
            scores_df = scores_df.pivot(index='feature_one', 
                                        columns='feature_two', 
                                        values=f"{distance_func.__name__}")

        return scores_df

    return inner


@feature_distances
def euclidean_distances(feature_one, feature_two, **kwargs):
    return euclidean(feature_one, feature_two)


@feature_distances
def euclidean_squared_distances(feature_one, feature_two, **kwargs):
    return euclidean_squared(feature_one, feature_two)


def get_sign_diffs(array_like):
    return np.diff(np.sign(np.diff(array_like)))


def get_optima_indices_simple(sign_diffs_arr, sign_change_val):
    """
    `sign_change_val` should be -2 for peaks or 2 for troughs.
    """
    return np.fromiter(
        (index[0] for index, x in np.ndenumerate(sign_diffs_arr) if x == sign_change_val),
        dtype=np.int32
        )


def get_peak_and_trough_indices_simple(array_like):
    sign_diffs_arr = get_sign_diffs(array_like)
    
    return (
        get_optima_indices_simple(sign_diffs_arr, -2), 
        get_optima_indices_simple(sign_diffs_arr, 2)
        )


def find_frequency_peaks_and_troughs(series=None, hist=None, bin_edges=None, num_bins=100):
    if hist is None or bin_edges is None:
        hist, bin_edges = np.histogram(series, bins=num_bins)

    peak_inds, trough_inds = get_peak_and_trough_indices_simple(hist)
    return bin_edges[peak_inds], bin_edges[trough_inds]


def get_optima_vals_simple(vals_arr, sign_change_val, sign_diffs_arr=None):
    if not sign_diffs_arr:
        sign_diffs_arr = get_sign_diffs(vals_arr)

    return vals_arr[get_optima_indices_simple(sign_diffs_arr, sign_change_val)]


def get_peak_vals_simple(array_like, sign_diffs_arr=None):
    return get_optima_vals_simple(array_like, -2, sign_diffs_arr)


def get_trough_vals_simple(array_like, sign_diffs_arr=None):
    return get_optima_vals_simple(array_like, 2, sign_diffs_arr)


def get_peak_and_trough_vals_simple(array_like):
    sign_diffs_arr = get_sign_diffs(array_like)
    
    return (
        get_peak_vals_simple(array_like, sign_diffs_arr), 
        get_trough_vals_simple(array_like, sign_diffs_arr)
        )