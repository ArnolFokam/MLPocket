from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


def drop(df: pd.DataFrame, columns: List[str]):
    """
    Drop columns

    :param df: dataframe to use
    :param columns: columns to drop

    :return: preprocessed dataframe
    """

    df = df.copy()

    for col in columns:
        if col in df.columns:
            del df[col]

    return df


def to(df: pd.DataFrame,
       input_cols: Union[str, List[str]],
       out_col: str,
       function,
       delete_input_col: bool = False):
    """
    map columns to output columns

    :param: df: dataframe to use
    :param: input_cols: input column to use
    :param: out_col: output column to create
    :param: function: lambda expression to map inputs
    :param: delete_input_col: should we delete the input column?

    :return: pre-processed dataframe
    """

    df = df.copy()

    # delete created columns to prevent error
    if out_col in df.columns and out_col not in input_cols:
        del df[out_col]

    # transform column
    df[out_col] = df.apply(lambda x: function(x[input_cols]), axis=1)

    # delete the output column after
    if delete_input_col:
        if isinstance(input_cols, str):
            del df[input_cols]
        elif isinstance(input_cols, (list, List)):
            for column in input_cols:
                del df[column]

    return df


def to_extract(df: pd.DataFrame,
               input_col: str,
               out_col: str,
               regex: str,
               delete_input_col: bool = False):
    """
    extract the pattern of interest

    :param df: dataframe to modify
    :param input_col: input column to use
    :param out_col: output column to create
    :param regex: pattern to match and extract
    :param delete_input_col:  should we delete the input column?

    :return: modified dataframe
    """

    df = df.copy()

    # delete created columns to prevent error
    if out_col in df.columns and out_col != input_col:
        del df[out_col]

    # transform column
    df[out_col] = df[input_col].str.extract(regex)

    # delete the output column after
    if delete_input_col:
        del df[input_col]

    return df


def to_length(
        df: pd.DataFrame,
        input_col: str,
        out_col: str,
        delete_input_col: bool = False):
    """
    create a column with length of each column

    :param df: dataframe to modify
    :param input_col: input column to use
    :param out_col: output column to create
    :param delete_input_col:  should we delete the input column?

    :return: modified dataframe
    """

    df = df.copy()

    # delete created columns to prevent error
    if out_col in df.columns and out_col != input_col:
        del df[out_col]

    # transform column
    df[out_col] = df[input_col].apply(lambda x: len(x))

    # delete the output column after
    if delete_input_col:
        del df[input_col]

    return df


def to_isnull(df: pd.DataFrame,
              input_col: str,
              out_col: str,
              delete_input_col: bool = False):
    """
    convert column map boolean value if null or not

    :param df: dataframe to modify
    :param input_col: input column to use
    :param out_col: output column to create
    :param delete_input_col:  should we delete the input column?

    :return: modified dataframe
    """

    df = df.copy()

    # delete created columns to prevent error
    if out_col in df.columns and out_col != input_col:
        del df[out_col]

    # transform column
    df[out_col] = df[input_col].apply(lambda x: 1 if pd.isnull(x) else 0)

    # delete the output column after
    if delete_input_col:
        del df[input_col]

    return df


def to_dummies(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        columns: List[str],
        delete_input_col: bool = False,):
    """
    Transform categorical values to dummy columns

    :param train_df: train dataframe to use
    :param test_df: test dataframe to use
    :param columns: columns to transform
    :param delete_input_col:  should we delete the input column?

    :return: preprocessed dataframe
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    all_columns = set(train_df.columns).union(test_df.columns)

    for column in columns:

        if column in all_columns:
            # convert to string
            train_df[column] = train_df[column].apply(lambda x: str(x))
            test_df[column] = test_df[column].apply(lambda x: str(x))

            # in case we have categories not present in the test dataframe
            good_cols = [column + '_' + i for i in train_df[column].unique() if i in test_df[column].unique()]

            # at the dummies
            train_df = pd.concat((train_df, pd.get_dummies(train_df[column], prefix=column)[good_cols]), axis=1)
            test_df = pd.concat((test_df, pd.get_dummies(test_df[column], prefix=column)[good_cols]), axis=1)

            if delete_input_col:
                del train_df[column]
                del test_df[column]

    return train_df, test_df


def to_fill_by_group(df: pd.DataFrame,
                     input_col: str,
                     out_col: str,
                     groups: Optional[List[str]] = None,
                     method: str = "mean",
                     delete_input_col: bool = False,
                     group_data: Optional[DataFrameGroupBy] = None):
    """
    convert column to mean, mode, max or min over group

    :param df: dataframe to modify
    :param input_col: input column to use
    :param out_col: output column to create
    :param groups: column to group field by
    :param method: method to fill values in groups (mean, mode, max, min)
    :param delete_input_col:  should we delete the input column?
    :param group_data: Group data to use if there is. Very useful when
                        we want to reuse the group in a test set

    :return: modified dataframe
    """

    assert method in ["mean", "max", "min", "mode"], "choose filling by group method"
    assert group_data or groups, "either insert the column to groupby or include group in parameter"

    df = df.copy()

    # delete created columns to prevent error
    if out_col in df.columns and out_col != input_col:
        del df[out_col]

    # transform column
    if group_data and isinstance(group_data, DataFrameGroupBy):
        data = group_data
    else:
        data = df.groupby(groups)[input_col]
    df[out_col] = data.transform(lambda x: x.fillna(
        x.mean() if method == "mean"
        else x.mode() if method == "mode"
        else x.min() if method == "min"
        else x.max()
    ))

    # delete the output column after
    if delete_input_col:
        del df[input_col]

    return df, data


def reduce_skewness(df: pd.DataFrame,
                    skew_threshold: int = 0.3, ):
    """
    Remove skewness in data for certain threshold

    :param df: dataframe that has columns to reduce skewness.
    :param skew_threshold: maximum skewness allowed

    :return: preprocessed data
    """

    df = df.copy()

    for col in df.columns:
        if np.abs(df[col].skew()) > skew_threshold:
            df[col] = np.log1p(df[col])

    return df
