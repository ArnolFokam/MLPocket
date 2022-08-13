from typing import List, Optional

import pandas as pd


def derive_column_in_dataframe(df: pd.DataFrame,
                               input_cols: List[str],
                               new_cols: List[str],
                               function,
                               delete_input_col: bool = False,
                               update_dataframe: bool = True):
    """
    Derive a column or a set of columns from a input or the set of inputs

    :param df: dataframe to operate on
    :param input_cols: columns where to get the inputs
    :param new_cols: output columns of the result
    :param function: function that maps a list of inputs (row of values from columns) to list of output
    :param delete_input_col: should we delete the input columns after the operation?
    :param update_dataframe: should we update the dataframe or just pass the update results?

    :return: the preprocessed dataframe, ouptut

    def derive(a):
        return [len(a[0]), a[1] + 50]

    derive_column(train,
        ['Name_Len', 'Pclass50'],
        ['Name', 'Pclass'], split_name, True)
    """

    df = df.copy()

    # delete created columns to prevent error
    for col in new_cols:
        if col in df.columns:
            del df[col]

    # derive the column
    tmp = df.apply(lambda x: function(x[input_cols].tolist()), axis=1)
    tmp = pd.DataFrame(tmp.tolist(), index=tmp.index)

    # update the original dataframe
    if update_dataframe:
        df[new_cols] = tmp

    # delete the output column after
    if delete_input_col:
        for col in input_cols:
            del df[col]

    return df, tmp


def derive_column_in_series(se: pd.Series,
                            function,
                            df: Optional[pd.DataFrame] = None,
                            new_cols: Optional[List[str]] = None,
                            update_dataframe: bool = True):
    """
    Derive a column or a set of columns from a pd.Series

    :param se: pd.Series that serves as input to get the output values
    :param function: function to map inputs to output
    :param df: dataframe to update
    :param new_cols: columns in the dataframe to update
    :param update_dataframe: should we update the dataframe or just pass the update results?

    :return: processed dataframe, output
    """

    # delete created columns to prevent error
    for col in new_cols:
        if col in df.columns:
            del df[col]

    # derive the column
    tmp = se.transform(lambda x: function(x))
    tmp = pd.DataFrame(tmp.tolist(), index=tmp.index)

    # update the original dataframe
    if update_dataframe and new_cols:
        df[new_cols] = tmp

    return df, tmp
