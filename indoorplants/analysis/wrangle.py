def get_feature_size_by_class(df, cls_col, features):
    """
    Given DataFrame, class column name, and feature column
    name, return:
    - pd.crosstab(df.cls_col, df.feature).stack()
    
    Works differently if multiple features passed.

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


def get_class_cnts_by_features_nulls(df, class_, features):
    """Retrieves, given a DataFrame, class column, and list
    of feature column names, a sort of crosstab-vector,
    where it's class col vs. all features, with a ratio
    of True:False (in class) included as well.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    class_ : str
    Column name for the class / target.

    features : iterable
    Iterable of column names of the feature(s).
    
    Return
    ------

    DataFrame with columns: [False, True, "tf_ratio], with
    column names in `features`, broken out by null-or-not,
    as indices.
    """ 
    groupbys = [df[[col, class_]
                   ].groupby([col, class_]
                   ).size(
                   ).unstack(
                   ).T.rename(columns={False: f"{col}_False",
                                       True: f"{col}_True"})
                for col in features]

    groupbys[0].index.name = ""
    _ahh = groupbys[0].join(groupbys[1:]).T.fillna(0)
    _ahh["tf_ratio"] = _ahh[True] / _ahh[False]
    return _ahh.sort_values("tf_ratio")


def get_null_stats(df):
    """I need a docstring"""
    nulls = df.isnull().sum()
    nulls = nulls.rename("cnt").to_frame()
    nulls["ratio"] = nulls / len(df)
    return nulls.sort_values("ratio", ascending=False)


def remove_cols_over_x_pcnt_null(df, x=.99, exclude=None):
    """I need a docstring"""
    nulls = get_null_stats(df)
    to_remove = nulls[nulls.ratio > x].index

    if isinstance(exclude, str):
        exclude = [exclude]
    if exclude:
        to_remove = list(filter(lambda x: x not in exclude, to_remove))

    return df.drop(to_remove, axis=1)


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