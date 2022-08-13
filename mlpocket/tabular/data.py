import warnings
from typing import List, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str,
              labels: Optional[List[str]] = None,
              d_format: Optional[str] = "csv",
              print_head: Optional[bool] = False,
              print_desc: Optional[bool] = False,
              **kwargs):
    """
    Loads tabular data from any format and separates the label from the inputs

    :param path: path to the data
    :param labels: labels to extract
    :param d_format: format in which the data is
    :param print_desc: print head
    :param print_head: print description

    :return: data in pandas format
    """

    if d_format == "csv":
        data = pd.read_csv(path, **kwargs)

        if print_head or print_desc:
            print(f"Data from {path}")

        # head rows
        if print_head:
            print(data.head(), "\n")

        # description of data
        if print_desc:
            print(data.describe(), "\n")
    else:
        raise ValueError("Unsupported data. Please, make sure your data is in either of these formats {csv}")

    if labels:
        X = data.drop(labels, axis=1)
        y = data[labels]

        return X, y, data
    else:
        return data


def conditional_probabilities(df: pd.DataFrame, col_a: str, col_b: str):
    """
    Calculate the conditional_probabilities of occurrences of values in col_a given values in col_b

    :param df:
    :param col_a:
    :param col_b:

    :return: return conditional probabilities as dictionary of dictionaries
    """

    vals_a = df[col_a].unique()
    if len(vals_a) > 20:
        warnings.warn("Unique values larger that 20 is not advisable")

    vals_b = df[col_b].unique()
    if len(vals_b) > 20:
        warnings.warn("Unique values larger that 20 is not advisable")

    c_probs = {}

    for val_a in vals_a:
        prob = {}
        for val_b in vals_b:
            # calculate p(b|a)
            prob[val_b] = len(df[df[col_a] == val_a].loc[df[col_b] == val_b]) / len(df[df[col_a] == val_a])
        c_probs[val_a] = prob

    return c_probs


def get_nan_stats(df: pd.DataFrame):
    """
    Prints statistics concerning NaN values in the dataframe

    :param df: dataframe to print statistics about

    :return: None
    """
    df = df.copy()

    print("\nPrinting percentage of NaN values per columns")
    for col in df.columns:
        tmp = df[col].apply(lambda x: 1 if pd.isnull(x) else 0)
        print(f"{col}: ", tmp.mean(), f"{tmp.sum()}/{len(df)} columns")

    return df[df.isna().any(axis=1)]


def plot_columns_dist(df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      sample: Optional[int] = None):
    """
    Plot distributions for each given column in the dataset

    :param sample: number of sample to plot the distribution from
    :param df: dataframe containing the columns
    :param columns: columns to use instead of all

    :return: None
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    new_df = df.select_dtypes(include=numerics)
    new_df = new_df.sample(sample if sample else len(df))
    columns_to_use = list(filter(lambda x: x in columns, new_df.columns) if columns else new_df.columns)

    fig, axes = plt.subplots(nrows=(len(columns_to_use) // 4) + 1, ncols=4)  # axes is 2d array (3x3)
    fig.tight_layout()
    fig.set_size_inches(25, 25)
    axes = axes.flatten()  # Convert axes to 1d array of length 9

    for ax, col in zip(axes[:len(columns_to_use)], columns_to_use):
        sns.histplot(new_df[col], ax=ax, kde=True, color='forestgreen')
        ax.set_title(col)

    return new_df


def plot_corr(df: pd.DataFrame,
              exclude: Optional[List[str]] = None, ):
    """
    Plot a correlation matrix of values in the dataframe

    :param df: dataframe to get the correlation matrx from
    :param exclude: columns to exclude

    :return: plot of correlation matrix
    """
    plt.figure(figsize=(20, 10))
    return sns.heatmap(df[
                           list(filter(lambda x: x not in exclude, df.columns))
                           if exclude else df.columns].corr(),
                       annot=True)


def reg_plot(features: pd.DataFrame,
             label: pd.Series,
             columns: Optional[List[str]] = None,
             normalize: bool = False):
    """
    Plot a correlation matrix of values in the dataframe

    :param features: dataframe to get the features of the regression line
    :param label: label to get the regression line from
    :param columns: columns to get the regression line from
    :param normalize: should we normalize the features first?

    :return: figure of the different regression line
    """

    assert label not in columns if columns else True, "label should not be in the features"

    features = features.loc[:, columns] if columns else features

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        features = pd.DataFrame(data=min_max_scaler.fit_transform(features), columns=features.columns)

    fig, axs = plt.subplots(nrows=(len(features.columns) // 4) + 1, ncols=4, figsize=(20, 10))
    axs = axs.flatten()

    for i, k in enumerate(features.columns):
        sns.regplot(y=label, x=features[k], ax=axs[i])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


def get_top_k_features(features: pd.DataFrame,
                       label: pd.Series,
                       k: str = 100,
                       **params):
    """
    Plot a correlation matrix of values in the dataframe

    :param features: features check
    :param label: column of label
    :param k: top k features to use

    :return: k best features
    """
    rf = RandomForestClassifier(**params)
    rf.fit(features, label)

    importance = pd.concat((
        pd.DataFrame(features.columns, columns=['variable']),
        pd.DataFrame(rf.feature_importances_, columns=['importance'])
    ), axis=1)

    k_best = importance.sort_values(by='importance', ascending=False)[:k]['variable']

    return importance, k_best.values
