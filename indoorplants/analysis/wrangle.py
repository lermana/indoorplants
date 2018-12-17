def get_binary_feature_size_by_class(df, class_, feature):
    """
    Given DataFrame, class column name, and feature column
    name, return what is I think, literally:
    - pd.crosstab(df.class_, df.feature).stack()
    Should this be a function then? Probably not. But
    hey, here it is.

    Parameters
    ----------
    
    df : pandas.DataFrame
        DataFrame on which this function will operate.

    class_ : str
    Column name for the class / target.

    feature : str
    Column name for the feature.
    

    Return
    ------

    DataFrame of size figures.
    """
    return df[[class_, feature]
              ].groupby([class_, feature]
              ).size(
              ).rename("cnt"
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
    nulls = df.isnull().sum
    nulls = nulls.rename("cnt").to_frame()
    nulls["ratio"] = nulls / len(df)
    return nulls


def remove_cols_over_x_pcnt_null(df, x=.99):
    """I need a docstring"""
    nulls = get_null_stats(df)
    to_remove = nulls[nulls.ratio > x].index
    return df.drop(to_remove, axis=1)
