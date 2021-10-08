import sys
import pandas as pd
import numpy as np
from collections import namedtuple
from utils import (
    get_path,
    split_data,
    get_dataset_helper,
    # make_two_datasets_discrete,
    # verify_dataset,
)

# For any dataset, X should be a DataFrame, y can be a Series (single label) or a DataFrame (multilabel)
dataset = namedtuple(
    "dataset", ("name", "X_train", "X_test", "y_train", "y_test", "is_multilabel")
)


classification_datasets = [
    "adult",
    "breast_cancer",
    "default_credit",
    "diabetic_retinopathy",
    "seismic_bumps",
    "statlog_australian_credit",
    "statlog_german_credit",
    "steel_plates_faults",
    "thoraric_surgery",
    "yeast",
]

regression_datasets = [
    "bike_sharing",
    "merck_ACT2",
    "merck_ACT4",
    "student_performance",
    "communities_and_crime",
    "parkinson_speech",
    "wine_quality",
    "concrete_compressive_strength",
    "qsar_aquatic_toxicity",
    "facebook_metrics",
    "sgemm_cpu_kernel_performance",
]


def get_datasets(names, test_size=0.2, random_state=0):
    datasets = []
    for name in names:
        if name == "adult":
            datasets.append(get_adult())
        elif name == "breast_cancer":
            datasets.append(get_breast_cancer_wisconsin(test_size, random_state))
        elif name == "default_credit":
            datasets.append(get_default_credit_card(test_size, random_state))
        elif name == "diabetic_retinopathy":
            datasets.append(get_diabetic_retinopathy(test_size, random_state))
        elif name == "seismic_bumps":
            datasets.append(get_seismic_bumps(test_size, random_state))
        elif name == "statlog_australian_credit":
            datasets.append(get_statlog_australian_credit(test_size, random_state))
        elif name == "statlog_german_credit":
            datasets.append(get_statlog_german_credit(test_size, random_state))
        elif name == "steel_plates_faults":
            datasets.append(get_steel_plates_faults(test_size, random_state))
        elif name == "thoraric_surgery":
            datasets.append(get_thoraric_surgery(test_size, random_state))
        elif name == "yeast":
            datasets.append(get_yeast(test_size, random_state))
        elif name == "bike_sharing":
            datasets.append(get_bike_sharing(test_size, random_state))
        elif name == "merck_ACT2":
            datasets.append(
                get_merck_molecular_activity_challenge_d1(test_size, random_state)
            )
        elif name == "merck_ACT4":
            datasets.append(
                get_merck_molecular_activity_challenge_d2(test_size, random_state)
            )

        elif name == "student_performance":
            datasets.append(get_student_performance(test_size, random_state))
        elif name == "communities_and_crime":
            datasets.append(get_communities_and_crime(test_size, random_state))
        elif name == "parkinson_speech":
            datasets.append(get_parkinson_speech(test_size, random_state))
        elif name == "wine_quality":
            datasets.append(get_wine_quality(test_size, random_state))
        elif name == "concrete_compressive_strength":
            datasets.append(get_concrete_compressive_strength(test_size, random_state))
        elif name == "qsar_aquatic_toxicity":
            datasets.append(get_qsar_aquatic_toxicity(test_size, random_state))
        elif name == "facebook_metrics":
            datasets.append(get_facebook_metrics(test_size, random_state))
        elif name == "sgemm_cpu_kernel_performance":
            datasets.append(get_sgemm_cpu_kernel_performance(test_size, random_state))
        else:
            print("unknown dataset name! " + name)
            sys.exit(0)
    return datasets


def get_adult():
    X_train, y_train = get_dataset_helper(
        path=get_path() + "classification/adult/",
        dataset_name="adult.data",
        missing_values=[" ?"],
        imputation="smart",
        label_indices=[14],
        categorical_encoding=False,
    )

    X_test, y_test = get_dataset_helper(
        path=get_path() + "classification/adult/",
        dataset_name="adult.test",
        missing_values=[" ?"],
        imputation="smart",
        label_indices=[14],
        categorical_encoding=False,
    )
    # make categorical, using values from both train and test
    c = len(X_train)
    both = pd.concat([X_train, X_test], axis=0, sort=False)
    both = pd.get_dummies(both)
    X_train = both.iloc[:c, :]
    X_test = both.iloc[c:, :]

    print(y_train, y_test)
    # both = pd.concat([X, y], axis=1, sort=False)
    # both = df.sort_values(by=['col1'])

    # X_train, X_test = make_two_datasets_discrete(True, X_train, X_test)
    # y_train, y_test = make_two_datasets_discrete(True, y_train, y_test)

    return dataset("adult", X_train, X_test, y_train, y_test, False)


get_adult()


def get_breast_cancer_wisconsin(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/breast_cancer_wisconsin/",
        dataset_name="wdbc.data",
        missing_values=["?"],
        index_col=0,
        label_indices=[0],
    )
    return dataset("breast_cancer", *split_data(X, y, test_size, random_state), False)


def get_default_credit_card(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/default_of_credit_card_clients/",
        dataset_name="default of credit card clients.xls",
        filetype="xls",
        header=0,
        index_col=0,
        label_indices=[23],
    )
    return dataset("default_credit", *split_data(X, y, test_size, random_state), False)


def get_diabetic_retinopathy(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/diabetic_retinopathy/",
        dataset_name="messidor_features.arff",
        filetype="arff",
        label_indices=[19],
    )
    return dataset(
        "diabetic_retinopathy", *split_data(X, y, test_size, random_state), False
    )


def get_seismic_bumps(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/seismic_bumps/",
        dataset_name="seismic-bumps.arff",
        filetype="arff",
        label_indices=[18],
    )
    return dataset("seismic_bumps", *split_data(X, y, test_size, random_state), False)


def get_statlog_australian_credit(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/statlog_australian_credit_approval/",
        dataset_name="australian.dat",
        csv_delimiter=" ",
        label_indices=[14],
    )
    return dataset(
        "statlog_australian_credit", *split_data(X, y, test_size, random_state), False
    )


def get_statlog_german_credit(test_size=0.2, random_state=0):
    # MAKES USE OF A COST MATRIX: it is worse to class a bad customer as good,
    # than a good customer as bad:
    #  1 = good, 2 = bad:
    #
    #   1     2
    # -----------
    # 1 | 0     1
    # 2 | 5     0
    # see sklearn.metrics.make_scorer
    X, y = get_dataset_helper(
        path=get_path() + "classification/statlog_german_credit_data/",
        dataset_name="german.data",
        csv_delimiter=" ",
        label_indices=[20],
    )
    return dataset(
        "statlog_german_credit", *split_data(X, y, test_size, random_state), False
    )


def get_steel_plates_faults(test_size=0.2, random_state=0):
    # multilabel dataset
    X, y = get_dataset_helper(
        path=get_path() + "classification/steel_plates_faults/",
        dataset_name="Faults.NNA",
        csv_delimiter="\t",
        label_indices=[27, 28, 29, 30, 31, 32, 33],
    )

    return dataset(
        "steel_plates_faults", *split_data(X, y, test_size, random_state), True
    )


def get_thoraric_surgery(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/thoraric_surgery_data/",
        dataset_name="ThoraricSurgery.arff",
        filetype="arff",
        label_indices=[16],
    )

    return dataset(
        "thoraric_surgery", *split_data(X, y, test_size, random_state), False
    )


def get_yeast(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "classification/yeast/",
        dataset_name="yeast.data",
        cols_to_drop=[0],  # first col is name, drop it
        label_indices=[9],
    )
    return dataset("yeast", *split_data(X, y, test_size, random_state), False)


def get_bike_sharing(test_size=0.2, random_state=0):
    X, y = get_dataset_helper(
        path=get_path() + "regression/bike_sharing/",
        dataset_name="hour.csv",
        index_col=0,
        label_indices=[15],
    )
    return dataset(
        "bike_sharing",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def _merck_helper(merck_file, test_size, random_state, name):
    path = get_path() + "regression/merck_molecular_activity_challenge/"
    with open(path + merck_file) as f:
        cols = f.readline().rstrip("\n").split(",")
        # Read the header line and get list of column names
        # Load the actual data, ignoring first column and using second column as targets.
        X = pd.DataFrame(
            np.loadtxt(
                path + merck_file,
                delimiter=",",
                usecols=range(2, len(cols)),
                skiprows=1,
                dtype=np.uint8,
            )
        )
        y = pd.Series(
            np.loadtxt(path + merck_file, delimiter=",", usecols=[1], skiprows=1)
        )
    return dataset(
        name, *split_data(X, y, test_size, random_state, method="random"), False
    )


def get_merck_molecular_activity_challenge_d1(test_size=0.2, random_state=0):
    return _merck_helper(
        "ACT2_competition_training.csv", test_size, random_state, "merck_ACT2"
    )


def get_merck_molecular_activity_challenge_d2(test_size=0.2, random_state=0):
    return _merck_helper(
        "ACT4_competition_training.csv", test_size, random_state, "merck_ACT4"
    )


def get_student_performance(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/student_performance/",
        dataset_name="merge.csv",
        # label_indices = [32, 52]  # use g1, g2 as attributes
        label_indices=[30, 31, 32, 50, 51, 52],  # try to guess g1 and g2 too
    )
    return dataset(
        "student_performance",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def get_communities_and_crime(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/communities_and_crime/",
        dataset_name="communities.data",
        label_indices=[127],
    )
    return dataset(
        "communities_and_crime",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def get_parkinson_speech(test_size=0.2, random_state=0.2):
    X_train, y_train = get_dataset_helper(
        path=get_path() + "regression/parkinson_speech/",
        dataset_name="train_data.txt",
        index_col=0,
        cols_to_drop=[27],  # drop class, should not be using it to train
        missing_values=[" ?"],
        label_indices=[26],
    )
    # TODO
    # X_test, y_test = get_dataset_helper(
    #    path=get_path() + "regression/parkinson_speech/",
    #    dataset_name="test_data.txt",
    #    cols_to_drop=[26],  # drop class
    #    label_indices
    #    index_col=0,
    # )
    return dataset(
        "parkinson_speech",
        *split_data(X_train, y_train, test_size, random_state, method="random"),
        False
    )


def get_wine_quality(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/wine_quality/",
        dataset_name="merge.csv",
        label_indices=[12],
        csv_delimiter=";",
    )
    return dataset(
        "wine_quality",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def get_concrete_compressive_strength(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/concrete_compressive_strength/",
        dataset_name="Concrete_Data.csv",
        label_indices=[8],
    )
    return dataset(
        "concrete_compressive_strength",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def get_qsar_aquatic_toxicity(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/qsar_aquatic_toxicity/",
        dataset_name="qsar_aquatic_toxicity.csv",
        csv_delimiter=";",
        label_indices=[8],
    )
    return dataset(
        "qsar_aquatic_toxicity",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def get_facebook_metrics(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/facebook_metrics/",
        dataset_name="dataset_Facebook.csv",
        csv_delimiter=";",
        imputation="smart",
        label_indices=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    return dataset(
        "facebook_metrics",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )


def get_sgemm_cpu_kernel_performance(test_size=0.2, random_state=0.2):
    X, y = get_dataset_helper(
        path=get_path() + "regression/sgemm_cpu_kernel_performance/",
        dataset_name="sgemm_product.csv",
        label_indices=[14, 15, 16, 17],
    )
    return dataset(
        "sgemm_cpu_kernel_performance",
        *split_data(X, y, test_size, random_state, method="random"),
        False
    )
