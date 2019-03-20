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

    pandas.DataFrame with columns: [False, True, "tf_ratio], with
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


def get_null_stats(df):
    """
    Function that will return count (absolute and relative)  of null 
    values in each column in `df`.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    Return
    ------

    pandas.DataFrame with `df.columns` as row index and `["cnt", "ratio"]` 
    (for absolute and relative counts, respectively) in column index.
    """
    nulls = df.isnull().sum()
    nulls = nulls.rename("cnt").to_frame()
    nulls["ratio"] = nulls / len(df)
    return nulls.sort_values("ratio", ascending=False)


def get_cols_over_x_pcnt_null(df, x=.99, exclude=None):
    """
    Function that will return all columns in `df` with over a certan 
    proportion of missing values.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    x : float, with value between 0 and 1 (default=.99)
        Proportion of rows allowed to be missing in a given column.

    exclude : str or list (default=None)
        Column name(s) to exclude from results.

    Return
    ------

    pandas.Index of columns with over `x` proportion missing values.
    """
    nulls = get_null_stats(df)

    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]

        nulls = nulls.loc[~nulls.index.isin(exclude)]

    return nulls[nulls.ratio > x].index


def remove_cols_over_x_pcnt_null(df, x=.99, exclude=None):
    """
    Function that will remove from `df` all columns with over an `x` proportion 
    of missing values. User can use `exclude` to prevent certain columns from 
    being removed.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    x : float, with value between 0 and 1 (default=.99)
        Proportion of rows allowed to be missing in a given column.

    exclude : str or list (default=None)
        Column name(s) to exclude from removal (i.e. for use when you know a
        given column has missing values but you'd like to keep it anyway).

    Return
    ------

    pandas.DataFrame with columns with columns with over an `x` proportion 
    of missing values removed.
    """
    to_remove = get_cols_over_x_pcnt_null(df, x=x, exclude=exclude)
    return df.drop(to_remove, axis=1)


def make_is_null_cols(df, x=.5, exclude=None, remove_originals=False):
    """
    Function that will make in `df` boolean re-representaions of all 
    columns with over an `x` proportion of missing values. I.e., allows 
    using simply whether the value of that column in that row is `null` 
    instead of taking the actual column value.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    x : float, with value between 0 and 1 (default=.99)
        Proportion of rows allowed to be missing in a given column.

    exclude : str or list (default=None)
        Column name(s) to exclude from consideration (i.e. for use when you know a
        given column has missing values but you'd like to keep it anyway).

    remove_originals : bool (default=False)
        Determines whether original columns are removed or left in `df`.

    Return
    ------

    `df`, including new, `is_null_...` columns.
    """
    null_cols = get_cols_over_x_pcnt_null(df, x=x, exclude=exclude)

    for col in null_cols:
        df["is_null_" + col] = df[col].isnull()

    if remove_originals:
        df = df.drop(null_cols, axis=1)

    return df


def get_cols_ratio_equal_val(df, val, ratio=1):
    """
    Useful for finding cols where a certain proportion of rows are equal
    to a particular value.
    """
    check = (df == val).sum()
    return check[check == (ratio * len(df))].index


def get_data_leak_cols_cls(df, cls_col,
                           threshold=.99,
                           dtypes=None,
                           drop_for_analysis=None,
                           join_for_analysis=None,
                           return_style="list"):
    """
    Makes use of `get_feature_size_by_class`. Currently works for categorical
    features.
    """

    # by default, columns of all `dtype` are selected (shouldn't happen if this works
    # only for categorical columns...)
    if dtypes is None:
        dtypes = [object, int, bool]

    df = df.select_dtypes(include=dtypes)

    # user can include additional data to be joined in; e.g. if separate `X` and `y`
    if join_for_analysis is not None:
        df = df.join(join_for_analysis)

    # user can specify certain columns to be excluded from analysis
    if drop_for_analysis:
        df = df.drop(drop_for_analysis, axis=1)

    # get size of each feature-value & class-value set; exclude `class_col` ...
    # ... from feature set to be examined
    feature_data = {
                        feature: get_feature_size_by_class(df, cls_col, feature)
                        for feature in filter(lambda c: c != cls_col, df.columns)
                    }

    # function to determine whether all feature values present against all class values
    is_not_same_num_vals_across_class = lambda x: len(
                                                    set(
                                                [
                                                    len(x.loc[x.index.levels[0][i]].index) 
                                                    for i in range(len(x.index.levels))
                                                ]
                                                    )) == 1

    # get ... is this erroneous? should the above func have `!=` ? kinda seems like it should
    missing_cols = [feature for feature in feature_data.keys() if 
                    is_not_same_num_vals_across_class(feature_data[feature])]

    is_feature_skewed_across_class = lambda x: len(x[x.ratio >= threshold].index) > 0

    skewed_cols = [feature for feature in feature_data.keys() if 
                   is_feature_skewed_across_class(feature_data[feature])]

    if return_style == "list":
        return list(set(set(missing_cols) | set(skewed_cols)))
    else:
        return {"missing": missing_cols, "skewed": skewed_cols}