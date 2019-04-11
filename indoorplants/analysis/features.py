import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def euclidean(feature_one, feature_two):
    """
    Returns Euclidean distance between the two passed features. Both
    are assumed to be of type `pd.Series` or uni-column `pd.DataFrame`.
    """
    return np.sqrt(
                (feature_one - feature_two
                   ).pow(2
                   ).sum())


def euclidean_squared(feature_one, feature_two):
    """
    Returns squared Euclidean distance, which accentuates dispartity moreso
    than its square root does, between the two passed features. Both are 
    assumed to be of type `pd.Series` or uni-column `pd.DataFrame`.
    """
    return (feature_one - feature_two
               ).pow(2
               ).sum()


def get_feature_iter_func(flat):
    """
    Helper function to determine what type of iteration to perform across two
    features. Takes `bool` as input.
    """
    if flat:
        return lambda feature_names: itertools.combinations(feature_names, 2)
    else:
        return lambda feature_names: itertools.product(feature_names, feature_names)


def normalize_min_max(df):
    """
    Uses min-max normalization to scale passed `pd.DataFrame` of values. Returns
    `pd.DataFrame` of transformed values with original row and column labels.
    """
    cols = df.columns
    inds = df.index

    arr = MinMaxScaler().fit_transform(df)
    return pd.DataFrame(arr, columns=cols, index=inds)


def feature_distances(distance_func):
    """
    Decorator for functions that calculate feature distances.

    `distance_func` is assumed to take as input, two features, likely in
    following the terms laid out in `euclidean`.

    This function will iterate through either the Cartesian Product (with
    `flat=True`) or all combinations of 2 (`flat=False`) of the specific
    features of the passed `DataFrame`. It will then return a nicely
    formatted `DataFrame` of results.

    Description of `inner` arguments, in order:
    - `pd.DataFrame` of data
    - `feature_names_list`, a list of  columns in `df` to compare
    - `flat`, boolean denoting result display style
    - `normalize_data`, which determines whether to min-max-scale data
    """
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
    """
    Wrapper allowing for running `euclidean` across many feature pairs.

    See `feature_distances` for more detail on how to pass arguments,
    and `euclidean` for details on metric calculation.
    """
    return euclidean(feature_one, feature_two)


@feature_distances
def euclidean_squared_distances(feature_one, feature_two, **kwargs):
    """
    Wrapper allowing for running `euclidean_squared` across many feature pairs.

    See `feature_distances` for more detail on how to pass arguments,
    and `euclidean_squared` for details on metric calculation.
    """
    return euclidean_squared(feature_one, feature_two)


def get_sign_diffs(array_like):
    """
    Given `array_like` (`np.array` or `pd.Series`), return an 
    array noting the change in direction from `array_like[row_index]` 
    to `array_like[row_index + 1]` for each row index in `array_like`.
    """
    return np.diff(np.sign(np.diff(array_like)))


def get_optima_indices_simple(sign_diffs_arr, sign_change_val):
    """
    Get inidices for peaks or troughs in passed `sign_diffs_arr`. Note that 
    `sign_change_val` should be -2 for peaks or 2 for troughs.
    """
    return np.flatnonzero(sign_diffs_arr == sign_change_val) + 1


def get_peak_and_trough_indices_simple(array_like):
    """
    Get indices of peaks and troughs in `array_like` using simple methodology 
    of looking for changes in direction. This method will likely benefit from 
    smoothing if data is "noisey."
    """
    sign_diffs_arr = get_sign_diffs(array_like)

    return (
        get_optima_indices_simple(sign_diffs_arr, -2),
        get_optima_indices_simple(sign_diffs_arr, 2)
        )


def get_bin_edge_inds_from_hist_inds(indices, bin_edges):
    """
    Given an array of indices (i.e. integer values - this was built with  
    the results of `get_optima_indices_simple` in mind) and an array of 
    histogram bin edges (assumed to be the second result of `np.histogram`) 
    return a `[num_bins, 2]` array, such that `bin_edges` is re-represented 
    to provide exactly two boundaries, with each pair corresponding to exactly 
    one value in `indices`. 
    """
    return bin_edges[np.vstack((indices, indices + 1)).T]


def find_frequency_peaks_and_troughs(series=None, hist=None, 
                                     bin_edges=None, bins=100):
    """
    Leverage functionality in `get_peak_and_trough_indices_simple` on 
    histogram data. If not passed (in `hist` and `bin_edges`), histogram 
    data is generated with `num_bins` bins.
    """
    if hist is None or bin_edges is None:
        hist, bin_edges = np.histogram(series, bins=bins)

    hist_peak_inds, hist_trough_inds = get_peak_and_trough_indices_simple(hist)

    return (
        get_bin_edge_inds_from_hist_inds(hist_peak_inds, bin_edges),
        get_bin_edge_inds_from_hist_inds(hist_trough_inds, bin_edges)
        )


def get_redundant_feature_pairs(feature_distance_series, threshold):
    """
    Expects results from a `feature_distances` func. Returns redundant 
    feature pairs, as determined by passed `threshold` for inter-feature 
    measurement.
    """
    return feature_distance_series[
                    feature_distance_series < threshold
                    ].reset_index(
                    ).iloc[:, :2
                    ].drop_duplicates()