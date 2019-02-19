import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_feature_size_by_class(df, cls_col, features):
    """
    Given DataFrame, class column name, and feature column
    name, return:
    - pd.crosstab(df.cls_col, df.feature).stack()
    
    Works differently from `pd.crosstab` if multiple features passed.

    Parameters
    ----------
    
    df : pandas.DataFrame
        DataFrame on which this function will operate.

    cls_col : str
    Column name for the class / target.

    features : str or list
    Column name(s) for the feature.
    

    Return
    ------

    DataFrame of size figures.
    """
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        raise TypeError("Must pass str or list for `features`.")

    to_group_by = [cls_col] + features

    return df[to_group_by
              ].groupby(to_group_by
              ).size(
              ).rename("ratio"
              ).to_frame(
              ) / len(df)


def get_class_cnts_by_features_nulls(df, class_col, features):
    """
    Note: this function may be partially broken.

    Retrieves, given: a DataFrame, class column, and list
    of feature column names; a sort of crosstab-vector,
    where it's `class_col` vs. all features, with a ratio
    of True:False (in class) included as well.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    class_col : str
    Column name for the class / target.

    features : iterable
    Iterable of column names of the feature(s).
    
    Return
    ------

    DataFrame with columns: [False, True, "tf_ratio], with
    column names in `features`, broken out by null-or-not,
    as indices.
    """ 
    groupbys = [df[[col, class_col]
                   ].groupby([col, class_col]
                   ).size(
                   ).unstack(
                   ).T.rename(columns={False: f"{col}_False",
                                       True: f"{col}_True"})
                for col in features]

    groupbys[0].index.name = ""
    _ahh = groupbys[0].join(groupbys[1:]).T.fillna(0)
    _ahh["tf_ratio"] = _ahh[True] / _ahh[False]
    return _ahh.sort_values("tf_ratio")

def drop_users_nully_obj_cols(users):
    """can this be generalized? does this kinda achieve what the above is trying and failing to do?"""
    obj = users.select_dtypes(include=object)
    obj_nulls = obj.isnull().sum()
    obj = obj.join(users.is_spam)

    obj_nice_table = obj.groupby("is_spam"
                        ).count(
                        ).T.join(obj_nulls.rename("null_cnt")
                        ).sort_values("null_cnt", ascending=False
                        ).join((obj_nulls[obj_nulls > 0] / len(users)
                                ).rename("null_ratio"))

    obj_nulls_90 = obj_nice_table[obj_nice_table.null_ratio >= .9].index

    return users.drop(list(filter(lambda c: c not in ("blog_url", "portfolio_completed"),
                                  obj_nulls_90)),
                      axis=1)


def get_null_stats(df):
    """I need a docstring"""
    nulls = df.isnull().sum()
    nulls = nulls.rename("cnt").to_frame()
    nulls["ratio"] = nulls / len(df)
    return nulls.sort_values("ratio", ascending=False)


def get_cols_over_x_pcnt_null(df, x=.99):
    nulls = get_null_stats(df)
    return nulls[nulls.ratio > x].index


def remove_cols_over_x_pcnt_null(df, x=.99, exclude=None):
    """I need a docstring"""
    to_remove = get_cols_over_x_pcnt_null(df, x=.99)

    if isinstance(exclude, str):
        exclude = [exclude]
    if exclude:
        to_remove = list(filter(lambda x: x not in exclude, to_remove))

    return df.drop(to_remove, axis=1)


def create_is_null_cols(df, null_threshold=.5, remove_originals=False, exclude=None):
    null_cols = get_cols_over_x_pcnt_null(df, null_threshold)

    if exclude is not None:
        null_cols = list(filter(lambda c: c not in exclude, null_cols))

    for col in null_cols:
        df["is_null_" + col] = df[col].isnull()

    if remove_originals:
        df = df.drop(null_cols, axis=1)

    return df


def get_cols_ratio_equal_val(df, val, ratio=1):
    """Useful for finding cols where a certain proportion of rows are equal
    to a particular value."""
    check = (df == val).sum()
    return check[check == (ratio * len(df))].index


def get_data_leak_cols_cls(df, cls_col,
                           threshold=.99,
                           dtypes=None,
                           drop_for_analysis=None,
                           join_for_analysis=None,
                           return_style="list"):
    """Currently works for categorical features."""
    if dtypes is None:
        dtypes = [object, int, bool]
    df = df.select_dtypes(include=dtypes)

    if join_for_analysis is not None:
        df = df.join(join_for_analysis)

    if drop_for_analysis:
        df = df.drop(drop_for_analysis, axis=1)

    feature_data = {feature: get_feature_size_by_class(df, cls_col, feature)
                    for feature in filter(lambda c: c != cls_col, df.columns)}

    is_not_same_num_vals_across_class = lambda x: len(set([len(x.loc[x.index.levels[0][i]].index) 
                                                          for i in range(len(x.index.levels))])
                                                     ) == 1

    missing_cols = [feature for feature in feature_data.keys() if 
                    is_not_same_num_vals_across_class(feature_data[feature])]
    
    is_feature_skewed_across_class = lambda x: len(x[x.ratio >= threshold].index) > 0
    
    skewed_cols = [feature for feature in feature_data.keys() if 
                   is_feature_skewed_across_class(feature_data[feature])]

    if return_style == "list":
        return list(set(set(missing_cols) | set(skewed_cols)))
    else:
        return {"missing": missing_cols, "skewed": skewed_cols}


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