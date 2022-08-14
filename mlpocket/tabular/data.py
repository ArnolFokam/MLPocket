import warnings
from typing import List, Optional, Dict, Tuple, Union

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


def load_data(path: str,
              labels: Optional[List[str]] = None,
              d_format: Optional[str] = "csv",
              **kwargs):
    """
    Loads tabular data from any format and separates the label from the inputs

    :param path: path to the data
    :param labels: labels to extract
    :param d_format: format in which the data is

    :return: data in pandas format
    """

    assert d_format in ["csv", "parquet"]

    if d_format == "csv":
        data = pd.read_csv(path, **kwargs)
    elif d_format == "parquet":
        data = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported data. Please, make sure pandas has a pd.read_{d_format}")

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


def get_nan_stats(dfs: Dict[str, pd.DataFrame],
                  print_nan_stats: bool = False,
                  plot_nan_stats: bool = False):
    """
    Prints statistics concerning NaN values in the dataframe

    :param dfs: dictionary of dataframes to print statistics about
    :param print_nan_stats: should we print nan stats?
    :param plot_nan_stats: should we plot nan stats?

    :return: None
    """

    if print_nan_stats:
        for name, df in dfs.items():
            print(f"\n\nPrinting percentage of NaN values per columns for {name} dataframe")
            for col in df.columns:
                tmp = df[col].apply(lambda x: 1 if pd.isnull(x) else 0)
                print(f"{col}: ", tmp.mean(), f"{tmp.sum()}/{len(df)} columns")

    if plot_nan_stats:
        # get missing values for all dataframes
        missing_values = pd.concat([df.isna().sum() for df in dfs.values()], axis=0) \
            .rename("missing values") \
            .reset_index() \
            .rename(columns={"index": "column"})

        # get percentage for each column in each dataframes
        missing = []
        for name, df in dfs.items():
            missing += [name] * len(df.columns)

        missing_values["data"] = missing

        # plot values
        _, ax = plt.subplots(figsize=(15, 15))
        sns.barplot(data=missing_values, y="column", x="missing values", hue="data", orient="h", ax=ax)


def plot_columns_dist(dfs: Dict[str, pd.DataFrame],
                      exclude: Optional[List[str]] = None,
                      grid_size: Tuple[int] = (7, 3)):
    """
    Plot distributions for each given column in the dataset

    :param dfs: dataframe containing the columns
    :param exclude: columns to use instead of all
    :param grid_size: gride size for the plot (rows, columns)

    :return: None
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # select the dataframes with the correct values to plot distribution
    for name, df in dfs.items():
        dfs[name] = df.select_dtypes(include=numerics)
        dfs[name]["df"] = name

    # get the different columns for all the dataframes
    total_columns = set.intersection(*[set(list(df.columns)) for df in dfs.values()])

    # get the intersection between the total columns and the specified columns.
    # If no columns were specified, take all total_columns
    columns_to_use = list(filter(lambda x: x not in exclude + ["df"], total_columns) if exclude else total_columns)

    plt.subplots(figsize=(25, 35))  # axes is 2d array (3x3)

    for i, column in enumerate(columns_to_use):
        plt.subplot(*grid_size, i + 1)
        tmp_df = pd.concat(dfs)
        if tmp_df[column].dtype in ['float16', 'float32', 'float64']:
            sns.histplot(data=tmp_df.reset_index(drop=True), x=column, hue="df")
        else:
            val_count = tmp_df[[column, "df"]].value_counts().rename("value_counts").reset_index()
            sns.barplot(data=val_count, x=column, y="value_counts", hue="df")
        plt.title(column)


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
                       annot=True,
                       cmap="RdYlGn", fmt='0.2f', vmin=-1, vmax=1, cbar=False)


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


def get_random_forest_top_k_features(features: pd.DataFrame,
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


def get_value_counts(df: pd.DataFrame,
                     column_name: str,
                     sort_by_column_name: bool = False):
    """
    Counts the different categories in the dataframe and return those values

    :param df: dataframe containing the column
    :param column_name: column to count the values from
    :param sort_by_column_name: should we sort the result?

    :return: value count for the column in the dataframe
    """

    value_count = df[column_name].value_counts() \
        .reset_index() \
        .rename(columns={column_name: "value count", "index": column_name}) \
        .set_index(column_name)

    value_count["percentage"] = df[column_name].value_counts(normalize=True) * 100
    value_count = value_count.reset_index()

    if sort_by_column_name:
        value_count = value_count.sort_values(column_name)

    return value_count


def compare_value_counts(dfs: Dict[str, pd.DataFrame],
                         column_name: str,
                         sort_by_column_name: bool = False):
    """
    Compare categories of a column across various dataframes

    :param dfs: list of dataframes
    :param column_name: column to compare the categories
    :param sort_by_column_name: should we sort a value?

    :return: return the table of values counts
    """
    value_counts: List[pd.DataFrame] = []

    for name, df in dfs.items():
        value_counts.append(get_value_counts(df, column_name, sort_by_column_name))
        value_counts[-1].rename(
            columns={"Value Count": f"{name}_value_count", "percentage": f"{name}_percentage"},
            inplace=True
        )

    val_count = pd.merge(*value_counts, on=column_name, how="outer")
    val_count = val_count.fillna(0)  # if the data is missing from a column, there is none so we fill with 0's

    final_val_count = val_count.drop(
        columns=[f"{name}_percentage" for name in dfs.keys()])  # avoid duplicating pie plots
    final_val_count.set_index(column_name).plot.pie(figsize=(12, 7),
                                                    legend=False,
                                                    ylabel="",
                                                    subplots=True,
                                                    title=dfs.keys())

    return val_count


def get_mi_scores(features: pd.DataFrame,
                  label: pd.Series,
                  discrete_features: Union[str, bool]):
    """
    Get mutual information scores for each features

    :param features: dataframe of features to consider
    :param label: labels
    :param discrete_features: should we consider discrete features as well?

    :return: return the MI score for each column
    """

    mi_scores = mutual_info_classif(features, label, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mean_label_per_feature(features: pd.DataFrame,
                                columns: List[str],
                                label: str):
    """
    Plot mean label value per column value in each columns

    :param features: dataframe to extract features from
    :param columns: columns to get the values
    :param label: labels to use

    :return: None
    """

    _ = plt.subplots(figsize=(20, 20))

    for i, column in enumerate(columns):
        # get mean value per column value
        temp_df = features.groupby([column])[label].mean()

        # get counts for each column
        temp_df_2 = features[column].value_counts()
        temp_df = temp_df[temp_df_2 > 50]

        plt.subplot(6, 2, i + 1)
        ax = sns.barplot(x=temp_df.index, y=temp_df.values)
        ax.set_ylim([0.15, 0.3])
        plt.ylabel(f"mean {label}")
        plt.xlabel(None)
        plt.title("Feature: " + column)
