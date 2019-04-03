import pandas as pd


def get_clean_df_index(df):
    if isinstance(df.index, pd.core.index.MultiIndex):
        return df.index.remove_unused_levels()
    else:
        return df.index


def get_feature_size_by_class(df, class_col, features, normalize="class"):
    """
    Given pandas.DataFrame, class column name, and feature column
    name, return:

        - pd.crosstab(df.class_col, df.feature).stack()

    Works differently from `pandas.crosstab` if multiple features passed.
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

    normalize : str, "class" or "all" or None (default="class")
        Whether to normalize counts. "class" means normalize counts within a 
        class value by the total counts in that group. "all" means normalize 
        by the length of `df`. `None` means do not normalize.

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

    if normalize == "all":
        to_return = (to_return / len(df)
                    ).rename(columns={"cnt": "ratio"})

    elif normalize == "class":
        denominators = to_return.groupby(level=0
                               ).sum(
                               ).rename(columns={"cnt": "cnt_cls_val"}
                               ).reset_index()

        to_return = to_return.reset_index(
                            ).merge(denominators, on=class_col
                            ).set_index(to_group_by)

        to_return = to_return.cnt.divide(to_return.cnt_cls_val
                                ).rename("ratio"
                                ).to_frame()

    elif normalize is None:
        pass

    else:
        raise ValueError("Improper value passed for `normalize`.")

    return to_return


def get_feature_sizes_dict(df, class_col, normalize="class"):
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
        kwarg in `pandas.Series.value_counts`, which is called on data that
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
        axis=1,
        sort=True
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


def get_cols_over_x_pcnt_null(df, x=.99, exclude=None, names_only=True):
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

    names_only : bool (default=True)
        Whether to return just column names (True) or whether to return 
        `get_null_stats` results for these columns (False).

    Return
    ------

    `pandas.Index` of columns with over `x` proportion missing values, or 
    `pandas.DataFrame` of `null` ratios.
    """
    nulls = get_null_stats(df)

    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]

        nulls = nulls.loc[~nulls.index.isin(exclude)]

    if names_only:
        return get_clean_df_index(nulls[nulls.ratio > x])
    else:
        return nulls[nulls.ratio > x]


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
    return check[check == (ratio * len(df))]


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
    return (
        counts.nunique() != 1  
            or counts.index.nunique() != 
               feature_size_by_class_df.index.levels[0].nunique()
            )


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
        booleans = (ratios >= threshold).any()

    return booleans.any()


def get_data_leak_cols_via_cls_counts(df, class_col, threshold=.99,
                                      dtypes=None, drop_for_analysis=None,
                                      join_for_analysis=None,
                                      return_style="dict"):
    """
    Function to assess whether features (categorical) in `df` are possibly 
    leaking data. This is achieved through through determining whether, for 
    a given feature:

        - every class value sees the same number of feature values
        - whether a disproportionate number of rows correspond to a certain 
          class value (where disproportionality is set using `threshold`)

    This latter insight is provided both at the feature and feature-value 
    level. See the `is_feature_overweighted_towards_class` function, or 
    below desctiption of `return_style` kwarg for more information.

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

    return_style : str (default="dict")
        If "dict" is passed, the following groups will be returned separately, 
        each under their own key in a `dict`:

            - not all class values see all feature values ("missing_vals")
            - there is a disproportionate concentration of a certain class 
              value at a certain feature ("overweight_feature")
            - there is a disproportionate concentration of a certain class 
              value at a certain feature value ("overweight_feature_value")

        Otherwise, a single `list` will be returned that contains all 
        column names.

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

    feature_sizes = get_feature_sizes_dict(df, class_col, normalize="class")

    filter_feature_data = lambda func, *args, **kwargs: [
                            k for k, v in feature_sizes.items()
                            if func(v, *args, **kwargs)
                        ]

    missing_vals = filter_feature_data(
                                is_feature_not_present_across_class
                                )

    overweight_feature = filter_feature_data(
                                is_feature_overweighted_towards_class, 
                                threshold,
                                feature_level=True
                                )

    overweight_feature_val = filter_feature_data(
                                is_feature_overweighted_towards_class,
                                threshold,
                                feature_level=False
                                )

    if return_style == "dict":
        return {
            "missing_vals": missing_vals,
            "overweight_feature": overweight_feature,
            "overweight_feature_val": overweight_feature_val
        }

    else:
        return list(
                    set(missing_vals)
                  | set(overweight_feature)
                  | set(overweight_feature_val)
                )


def get_missing_around_class_stats(null_counts, names_only=True):
    """
    Given results from `get_class_cnts_by_many_features_nulls`, get
    features whose values are entirely `null` at a given class 
    value.

    Parameters
    ----------

    null_counts : pandas.DataFrame
        Results from `get_class_cnts_by_many_features_nulls`.

    names_only : bool (default=True)
        Whether to return just names of features who are `null` for 
        a given class value (True) or whether to return full `null_counts` 
        table for these features (False).

    Return
    ------

    Either `pandas.DataFrame` or `list`, representing `null_counts` filtered 
    to features that are entirely `null` at a given class value.
    """
    missing_vals = null_counts[null_counts.isnull().any(axis=1)
                              ].dropna(how="all")

    if names_only:
        missing_vals = list(
                        get_clean_df_index(missing_vals
                            ).levels[0]
                        )

    return missing_vals


def get_null_count_spreads(null_counts, exclude=None):
    """
    Given results from `get_class_cnts_by_many_features_nulls`, get absolute 
    values of differences, for each class value, between the proportion of rows 
    taking that class value when a given feature is `null` and when a given 
    feature is `not-null`.

    Parameters
    ----------

    null_counts : pandas.DataFrame
        Results from `get_class_cnts_by_many_features_nulls`.

    exclude : list-like (default=None)
        Features to remove from returned results.

    Return
    ------

    `pandas.DataFrame` with features as indices, class values as columns, and 
    `not-null` absolute differences (as described above) as values.
    """
    if exclude is None:
        exclude = []

    null_counts.index = null_counts.index.rename(["feature", "is_null"])
    null_counts = null_counts.reset_index()

    null_counts = null_counts.loc[
                            (~null_counts.feature.isin(exclude))
                          & (null_counts.notnull().all(axis=1))
                         ]

    to_return = null_counts.groupby("feature"
                          ).diff(
                          ).set_index(null_counts.feature
                          ).dropna(
                          ).abs(
                          ).iloc[:, 0]

    to_return.columns = ["spread"]
    return to_return


def get_over_threshold_columns(spreads, threshold, names_only=True):
    """
    Given results from `get_null_count_spreads`, get features whose 
    spreads exceed `threshold`.

    Parameters
    ----------

    spreads : pandas.DataFrame
        Results from `get_null_count_spreads`.

    threshold : float, with value between 0 and 1 (default=.99)
        Measure of maximum acceptibility for spread between the proportion 
        of rows with a given class value when a given feature is `null`, vs. 
        when it's `not-null`. Spread is the difference between these two 
        proportions.

    names_only : bool (default=True)
        Whether to return just names of features that exceed `threshold`, or 
        the `get_null_count_spreads` results for these features.

    Return
    ------

    Either `pandas.DataFrame` or `list`.
    """
    over_threshold = spreads[spreads > threshold].dropna()

    if names_only:
        spreads = list(
                    spreads[spreads > threshold
                          ].dropna(
                          ).index
                )

    return spreads


def get_data_leak_cols_via_nulls(df, class_col, threshold=.5, dtypes=float,
                                 drop_for_analysis=None, join_for_analysis=None,
                                 return_style="dict"):
    """
    Function to assess whether features in `df` are possibly leaking data. 
    This is achieved in a simple fashion, through considering the presence of 
    `null` values and how that varies across different class values. More 
    specically, this function provides:

        - whether a feature is entirely `null` over a certain class value
        - whether there's a unreasonably high spread between the `null` and 
          `not-null` counts at a given class value, where the maximim spread 
          considered reasonable is set using `threshold`

    Note that these two checks are mutually exclusive, and also that 
    features whose values are not missing at all will be excluded from the
    results.

    Parameters
    ----------

    df : pandas.DataFrame
        DataFrame on which this function will operate. These features can 
        be categorical or continuous.

    class_col : str
        Name of class column.

    threshold : float, with value between 0 and 1 (default=.99)
        Measure of maximum acceptibility for spread between the proportion 
        of rows with a given class value when a given feature is `null`, vs. 
        when it's `not-null`. Spread is the difference between these two 
        proportions.

    dtypes : type, or list[type] (default=float)
        Specific column type(s) to limit analysis to.

    drop_for_analysis : str or list (default=None)
        Column name(s) to exclude from consideration.

    join_for_analysis : pd.Series or pd.DataFrame (default=None)
        Additional data to be joined in to `df; e.g. if separate `X` and `y`.
        `class_col` will be looked up post-join, if there is a join.

    return_style : str (default="dict")
        If "dict" is passed, the analysis results will be returned separately, 
        each under their own key in a `dict`:

            - features for which one or more class values see only `null`
              ("missing_vals")
            - features for which spread, as defined above, exceeds `threshold` 
              ("over_threshold")

        Otherwise, a single `list` will be returned that contains all 
        column names.

    Return
    ------

    Columns with potential data leak issues, formatted as per `return_style`.
    """
    if drop_for_analysis:
        df = df.drop(drop_for_analysis, axis=1)

    df = df.select_dtypes(include=dtypes)

    if join_for_analysis is not None:
        df = df.join(join_for_analysis)

    null_counts = get_class_cnts_by_many_features_nulls(
                                df,
                                class_col,
                                filter(lambda c: c != class_col, df.columns)
                                )

    missing_vals = get_missing_around_class_stats(null_counts)

    over_threshold = get_over_threshold_columns(
                        spreads=get_null_count_spreads(
                                        null_counts=null_counts, 
                                        exclude=missing_vals
                                        ),
                        threshold=threshold
                        )

    if return_style == "dict":
        return {
                "missing_vals": missing_vals,
                "over_threshold": over_threshold
            }

    else:
        return missing_vals + over_threshold