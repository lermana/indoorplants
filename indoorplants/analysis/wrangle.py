def get_feature_size_by_class(df, cls_col, features, normalize=True):
    """
    Given pandas.DataFrame, class column name, and feature column
    name, return:
    - pd.crosstab(df.cls_col, df.feature).stack()

    Works differently from `pd.crosstab` if multiple features passed.
    Note that this function makes the most sense for categorical 
    features.

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

    pandas.DataFrame of size figures.
    """
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        raise TypeError("Must pass str or list for `features`.")

    to_group_by = [cls_col] + features

    to_return = df[to_group_by
                  ].groupby(to_group_by
                  ).size(
                  ).rename("cnt"
                  ).to_frame()

    if normalize:
        to_return = (to_return / len(df)
                    ).rename(columns={"cnt": "ratio"})

    return to_return


def get_class_cnts_by_feature_null(df, class_col, feature, normalize=True):
    """
    Break out class fequencies (in `df[class_col]`) by whether or not 
    `df[feature]` is null.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    cls_col : str
        Column name for the class / target.

    feature : str
        Column name for the feature.

    normalize : bool (default=True)
        Whether or not to normalize class counts by number of rows in 
        the respective feature is: [null / non-null] query. I.e. the
        value for `normalize` is passed straight to the `normalize`
        kwarg in `pd.Series.value_counts`, which is called on data that
        is filtered for either `df[feature].isnull()` of `df[feature].notnull()`.

    Return
    ------

    pandas.DataFrame of class counts, broken out by whether or not 
    `df[feature]` is null.
    """
    null = df.loc[df[feature].isnull(), class_col
                 ].value_counts(normalize=normalize
                 ).rename("null"
                 ).to_frame()

    not_null = df.loc[df[feature].notnull(), class_col
                     ].value_counts(normalize=normalize
                     ).rename("not_null")

    return pd.concat({feature: null.join(not_null)}, axis=1)


def get_class_cnts_by_many_features_nulls(df, class_col, features_list, normalize=True):
    """
    Wrapper for `get_class_cnts_by_feature_null`, with only difference being that, in 
    this function, users passes a `list` of features, as opposed to a single feature.
    """ 
    return pd.concat(
        [
            get_class_cnts_by_feature_null(df, class_col, f, normalize=normalize)
            for f in features_list
        ],
        axis=1
    )


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
        Column name(s) to exclude from consideration (i.e. for use when you 
        know a given column has missing values but you'd like to keep it 
        anyway).

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


def get_data_leak_cols_cls(df, cls_col, threshold=.99, dtypes=None,
                           drop_for_analysis=None, join_for_analysis=None,
                           return_style="list"):
    """
    Makes use of `get_feature_size_by_class`. Currently works for categorical
    features in classification problems.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    cls_col: str
        Name of class column.

    threshold : float, with value between 0 and 1 (default=.99)
        Proportion of rows allowed to be missing in a given column.

    dtypes: type, or list[type] (default=None)
        Specific column type(s) to limit analysis to. Defaults to all types except
        `float` if no value passed.

    drop_for_analysis : str or list (default=None)
        Column name(s) to exclude from consideration.

    join_for_analysis : pd.Series or pd.DataFrame (default=None)
        Additional data to be joined in to `df; e.g. if separate `X` and `y`.
        `cls_col` will be looked up post-join, if there is a join.

    return_style: str (default="list")
        If "list" is passed, a `list` containing the names of both missing and skewed
        columns will be returned. Otherwise, these two groups will be returned
        separately, each under their own key in a `dict`.

    Return
    ------

    `df`, including new, `is_null_...` columns.
    """
    if dtypes is None:
        kwargs = {"exclude": float}
    else:
        kwargs = {"include": dtypes}

    df = df.select_dtypes(**kwargs)

    if join_for_analysis is not None:
        df = df.join(join_for_analysis)

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