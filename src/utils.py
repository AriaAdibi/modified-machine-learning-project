import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
)
from sklearn import model_selection
from scipy.io import arff
import os.path


def get_path():
    with open("path_to_data.txt") as f:
        return f.readlines()[0].strip()


def split_data(X, y, test_size, random_state, method="stratified"):
    """ Returns a split dataset.
        If the problem is classification, you can choose:
            'stratified' for a stratified split
            'random' for a random split. best for balanced datasets.
        If the problem is stratified, you can either choose:
            'sorted strat' for a sorted stratification, (only works when y is a Series)
            'random' for random stratification.
    """
    if method == "stratified":
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    # sorted stratification, https://scottclowe.com/2016-03-19-stratified-regression-partitions/
    # can only work if the target variable is sortable, aka is a Series.
    elif method == "sorted_strat":
        pass
        # TODO
        # both = pd.concat([X, y], axis=1, sort=False)
        # both = df.sort_values(by=['col1'])
        # print("REGRESSION")
        # print(both)
    elif method == "random":
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=test_size, random_state=0
        )
    else:
        print("unknown dataset split method:", method)
        sys.exit(0)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Normalizing the data
    features_to_scale = X_train.select_dtypes(exclude=["category"]).columns
    scaler = StandardScaler().fit(X_train[features_to_scale])

    scaled_train = pd.DataFrame(scaler.transform(X_train[features_to_scale]))
    scaled_train.columns = features_to_scale
    scaled_test = pd.DataFrame(scaler.transform(X_test[features_to_scale]))
    scaled_test.columns = features_to_scale

    for c in scaled_train.columns:
        X_train.loc[:, c] = scaled_train.loc[:, c]
    for c in scaled_test.columns:
        X_test.loc[:, c] = scaled_test.loc[:, c]

    return X_train, X_test, y_train, y_test


def get_header(path):
    with open(path) as f:
        return [l.strip() for l in f.readlines()]


def get_dataset_helper(
    path,
    dataset_name,
    filetype="csv",
    header=None,
    missing_values=None,
    imputation=None,
    index_col=False,
    label_indices=None,
    cols_to_drop=None,
    csv_delimiter=None,
    categorical_encoding=True,
):
    """
    The main function for reading in datasets.
    path:                       the path of the dataset folder
    dataset_name:               the name of the dataset
    filetype:                   the file extension: can be csv, xls, or arff
    header:                     list of column names. leave as None if the folder contains a header.txt
    missing_values:             [missing values] array of values that should be intepreted as missing values
    imputation:                 what is done with missing values.
                                    "remove" removes the sample with missing values
                                    "mean" replaces the value with the mean of the column
    index_col:                  If there is an column that holds the index of each sample, the index of that column
    label_indices:              which columns are the targets for classification/regression. This happens before columns are dropped.
    cols_to_drop:               which columns should be dropped.
    csv_delimiter:              for csv files, if the delimiter is something other than "," which pandas doesn't detect automatically.
    categorical_encoding:       Whether or not categorical values should be made into dummy values. Set to false if you a given an external test set
    """

    # handle header
    if header is None and os.path.exists(path + "header.txt"):
        header = get_header(path + "header.txt")

    # read file in
    if filetype == "csv":
        data = pd.read_csv(
            path + dataset_name,
            header=None,
            names=header,
            index_col=index_col,
            na_values=missing_values,
            delimiter=csv_delimiter,
        )
    elif filetype == "xls":
        data = pd.read_excel(
            path + dataset_name,
            encoding="utf-8",
            header=header,
            index_col=index_col,
            na_values=missing_values,
        )
    elif filetype == "arff":
        with open(path + dataset_name) as f:
            # changed to loadarff from scipy to support NaN values
            data = pd.DataFrame(arff.loadarff(f)[0])
            if header is not None:
                data.columns = header
            if missing_values is not None:
                print(
                    "Custom missing values is not (yet) supported by arff input,"
                    " you might have issues in your imported dataset."
                )

    else:
        print("unsupported datatype")
        sys.exit(0)

    # handle missing values
    if imputation is not None:
        if imputation == "remove":
            data = data.dropna(axis=0)  # drop rows with NA values
        elif imputation == "mean":
            data = data.fillna(data.mean(axis=0).to_dict(), axis=0)
        elif imputation == "smart":
            # if there is less than 10% of missing values in a column, replace it with the mean (mode for categorical data) value:
            for c in data.columns:
                missing = data[c].isnull().sum() / len(data[c])
                if missing != 0:
                    if missing < 0.010:
                        if data[c].dtype == "object":
                            data[c] = data[c].fillna(data[c].mode()[0])
                        else:
                            data[c] = data[c].fillna(data[c].mean())
                    else:  # else, drop any rows which have missing in that column
                        data = data.dropna(how="any", subset=[c])
        data = data.reset_index(drop=True)

    # get column name of columns needed to be dropped, will be dropped after X/y split is done
    # this is done now to not mess up for labels indices
    if cols_to_drop is not None:
        cols_to_drop = data.columns[cols_to_drop]

    # make X and y
    if label_indices is None:  # if not test column provided
        X = data
        y = None
    else:
        X = data.drop(data.columns[label_indices], axis=1)
        y = data[data.columns[label_indices]]

    # handle dropping of columns
    if cols_to_drop is not None:
        for c in cols_to_drop:
            if c in X.columns:
                X = X.drop(columns=[c])
            if c in y.columns:
                y = y.drop(columns=[c])

    # get series, so this dataset is identified as mono-label
    if label_indices is not None and len(label_indices) == 1:
        y = y.T.squeeze()

    # encoding of categories
    if y is not None:
        if isinstance(y, pd.Series):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=y.name)
        elif any(y.dtypes == "object"):  # multilabel case
            for i in y.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                y.loc[:, i] = pd.Series(
                    le.fit_transform(y.loc[:, i]), name=y.loc[:, i].name
                )

    if categorical_encoding is True:
        X = pd.get_dummies(X)

    # verify_dataset(X, y)
    return X, y


def is_imbalanced(y):
    """Determines whether or not a dataset is imbalanced with respect to its classes."""
    # dataset is imbalanced if a class represents 15% less than 1/n
    def is_imbalanced_series(y: pd.Series):  # y should be a pd.Series
        target_counts = y.value_counts()
        return any(
            [c / y.count() < (1 / len(target_counts) - 0.15) for c in target_counts]
        )

    # if y is a series, check if any labels are imbalanced.
    if isinstance(y, pd.Series):
        return is_imbalanced_series(y)
    # else, if y is a DataFrame: for every label type, check if they are imbalanced
    else:
        return any([is_imbalanced_series(y[col]) for col in y])


def verify_dataset(X, y=None):  # for debugging
    if type(X) == np.ndarray:
        X = pd.DataFrame(X)
    print("X NaN presence: ", X.isnull().values.any())
    print(X)
    print(X.dtypes)
    print("\n\n\n")
    if y is not None:
        if type(y) == np.ndarray:
            y = pd.DataFrame(y)
        print(type(y))
        print("y NaN presence: ", y.isnull().values.any())
        print(y)
        print(y.dtypes)
