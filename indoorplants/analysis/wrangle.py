import pandas as pd

def get_feature_size_by_class(df, class_col, features, normalize=True):
    """
    Given pandas.DataFrame, class column name, and feature column
    name, return:

        - pd.crosstab(df.class_col, df.feature).stack()

    Works differently from `pd.crosstab` if multiple features passed.
    Note that this function makes the most sense for categorical 
    features.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate.

    class_col : str
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

    to_group_by = [class_col] + features

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

    class_col : str
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


def get_class_cnts_by_many_features_nulls(df, class_col, features_list,
                                          normalize=True):
    """
    Wrapper for `get_class_cnts_by_feature_null`, with only difference being 
    that, in this function, users passes a `list` of features, as opposed to 
    a single feature.
    """ 
    return pd.concat(
        [
            get_class_cnts_by_feature_null(df, class_col, f, normalize=normalize)
            for f in features_list
        ],
        axis=1
    ).T


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


def get_feature_sizes_dict(df, class_col, normalize=True):
    """
    Helper function to get size of each feature-value & class-value set,
    with `class_col` excluded from feature set to be examined.
    """
    return {
                feature: get_feature_size_by_class(df,
                                                   class_col,
                                                   feature,
                                                   normalize=normalize)

                for feature in filter(lambda c: c != class_col, df.columns)
    }


def is_feature_not_present_across_class(feature_size_by_class_df):
    """
    Helper function to see whether all subsections in `feature_size_by_class_df`, 
    based on slicing with different values in first-level rwo index, have same 
    number of values in second-level row index.

    Note that `feature_size_by_class_df` then must obviously have at least two 
    row-index levels.

    The intended use case is for handling results from `get_feature_size_by_class`,
    where first-level row index corresponds to class values, and second-level
    row index corresponds to feature values.

    The idea is to confirm that all feature values are present across all class 
    values. If this is not true, feature is either highly predictive or leaking
    data.

    Feature in `feature_size_by_class_df` should be discrete, i.e. of a finite 
    set of values.

    Parameters
    ----------

    feature_size_by_class_df : pandas.DataFrame
        DataFrame on which this function will operate, which must have at least
        two row-index levels.

    Return
    ------

    `bool` indicating whether all class values see all feature values.
    """
    index_vals = feature_size_by_class_df.reset_index().iloc[:, :2]
    counts = index_vals.groupby(index_vals.columns[0]).size()
    return counts.nunique() != 1


def is_feature_overweighted_towards_class(feature_size_by_class_df,
                                          threshold=.99,
                                          feature_level=True):
    """
    The intended use case is for seeing whether a categorical feature very 
    strongly points towards a particular class value, as, if this is true, 
    the feature is either highly predictive or leaking data.

    Parameters
    ----------

    feature_size_by_class_df : pandas.DataFrame
        DataFrame on which this function will operate. This is presumed to 
        be a result from the function `get_feature_size_by_class`. Additionally, 
        the feature represented in `feature_size_by_class_df` should be discrete, 
        i.e. comes from a finite set of values.

    threshold : float, with value between 0 and 1 (default=.99)
        Proportion of rows allowed that are allowed to be taken up by single
        `class_col` & `feature` value grouping.

    feature_level : bool (default=True)
        Whether to perform at the feature level, or feature-value level.

    Return
    ------

    `bool` indicating whether all class values see all feature values.
    """
    ratios = feature_size_by_class_df.unstack(0).ratio

    if feature_level:
        booleans = ratios.sum() >= threshold
    else:
        booleans = ratios >= threshold

    return booleans.any()


def get_data_leak_cols_cat(df, class_col, threshold=.99, dtypes=None,
                           drop_for_analysis=None, join_for_analysis=None,
                           overweight_level="feature", return_style="dict"):
    """
    Function to assess whether features (categorical) in `df` are possibly 
    leaking data. This is achieved through through determining whether, for 
    a given feature:

        - every class value sees the same number of feature values
        - whether a disproportionate number of rows correspond to a certain 
          class value (where disproportionate-ness is set using `threshold`)

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate - features should be 
        categorical.

    class_col: str
        Name of class column.

    threshold : float, with value between 0 and 1 (default=.99)
        Proportion of rows allowed that are allowed to be taken up by single
        `class_col` & `feature` or `class_col` & `feature`-value grouping.

    dtypes : type, or list[type] (default=None)
        Specific column type(s) to limit analysis to. Defaults to all types 
        except `float` if no value passed.

    drop_for_analysis : str or list (default=None)
        Column name(s) to exclude from consideration.

    join_for_analysis : pd.Series or pd.DataFrame (default=None)
        Additional data to be joined in to `df; e.g. if separate `X` and `y`.
        `class_col` will be looked up post-join, if there is a join.

    overweight_level : str (default="feature")
        Determines whether to check for `threshold` exceedance. If given value 
        "feature" will check at the feature level, and otherwise will check 
        at the feature-value level.

    return_style : str (default="list")
        If "list" is passed, a single `list` will be returned that contains 
        the names of both features for which:

            - not all class values see all feature values ("missing_vals")
            - there is a disproportionate concentration of a certain class 
              value at a certain feature or feature value ("overweight")

        Otherwise, these two groups will be returned separately, each under 
        their own key in a `dict`.

    Return
    ------

    Columns with potential data leak issues, formatted as per `return_style`.
    """
    if dtypes is None:
        kwargs = {"exclude": float}
    else:
        kwargs = {"include": dtypes}

    if drop_for_analysis:
        df = df.drop(drop_for_analysis, axis=1)

    df = df.select_dtypes(**kwargs)

    if join_for_analysis is not None:
        df = df.join(join_for_analysis)

    feature_sizes = get_feature_sizes_dict(df, class_col, normalize=True)

    filter_feature_data = lambda func, *args, **kwargs: [
                            k for k, v in feature_sizes.items()
                            if func(v, *args, **kwargs)
                        ]

    cols_missing_vals = filter_feature_data(
                                is_feature_not_present_across_class
                                )
    
    cols_over_threshold = filter_feature_data(
                                is_feature_overweighted_towards_class, 
                                threshold,
                                feature_level=overweight_level == "feature"
                                )

    if return_style == "list":
        return list(set(cols_missing_vals) | set(cols_over_threshold))
    else:
        return {"missing_vals": cols_missing_vals, "overweight": cols_over_threshold}