import pandas as pd
import os
from get_data import get_datasets, classification_datasets, regression_datasets
from class_model_sel import logr, gnb, knn, svc, gboostc, adaboostc, rfc, dtc, mlpc
from reg_model_sel import linr, gpr, svr, gboostr, adaboostr, rfr, dtr, mlpr
from pickling import save_model, load_model, save_datasets, load_datasets
from utils import get_path

from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    r2_score,
    mean_squared_error,
    max_error,
    explained_variance_score,
)

# these contain all the algos, including those who take too long to compute
classification_algos = [logr, gnb, knn, svc, gboostc, adaboostc, rfc, dtc, mlpc]
regression_algos = [linr, gpr, svr, gboostr, adaboostr, rfr, dtr, mlpr]

# these remove the algos who take too long to compute
classification_algos = [logr, gnb, knn, adaboostc, rfc, dtc, mlpc]
regression_algos = [linr, svr, gboostr, adaboostr, rfr, dtr, mlpr]


def measure_performance(y_pred, y_test, is_multilabel, is_classification):
    # handle multilabel case
    if is_multilabel:
        average = "micro"
    else:
        average = "binary"
    if is_classification:
        print("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)), "\n")
        print(
            "F1_score: {0:.3f}".format(
                fbeta_score(y_test, y_pred, beta=1, average=average)
            ),
            "\n",
        )
        print("Classification report")
        print(classification_report(y_test, y_pred), "\n")
        print("Confusion matrix")
        if is_multilabel:
            print(multilabel_confusion_matrix(y_test, y_pred), "\n")
        else:
            print(confusion_matrix(y_test, y_pred), "\n")
    else:
        print("R2 score: {0:.3f}".format(r2_score(y_test, y_pred)), "\n")
        print(
            "mean squared error score: {0:.3f}".format(
                mean_squared_error(y_test, y_pred), "\n"
            )
        )
        print("max error score: {0:.3f}".format(max_error(y_test, y_pred)), "\n")
        print(
            "explained variance score: {0:.3f}".format(
                explained_variance_score(y_test, y_pred)
            ),
            "\n",
        )


def fit_and_save_models(
    datasets, algos, random_state=0, models_dir="models", is_classification=True
):
    for d in datasets:
        for f in algos:
            # hyperparameter search for f algorithm
            print("Training " + f.__name__ + " on " + d.name + " dataset:")
            # handle multilabel case
            if isinstance(d.y_train, pd.DataFrame):
                scoring = "f1_micro"
            elif is_classification is True:
                scoring = "f1"
            else:
                scoring = "r2"

            model = f(
                d.X_train,
                d.y_train,
                random_state=random_state,
                scoring=scoring,
                is_multilabel=d.is_multilabel,
            )
            save_model(
                f.__name__, d.name, d.is_multilabel, model, random_state, models_dir
            )


def load_models_and_predict(
    datasets, algos, random_state, models_dir, is_classification
):

    for d in datasets:
        for f in algos:
            a = f.__name__
            model, is_multilabel, _ = load_model(models_dir, a, d.name)
            y_pred, score = model.predict(d.X_test), model.score(d.X_test, d.y_test)
            print(a + " on " + d.name + " score: " + str(score))
            measure_performance(y_pred, d.y_test, is_multilabel, is_classification)
            # maybe store the predictions on a file?
            # do some graphing here with predict / score...
            pass


def get_pickled_datasets():
    # classification
    if os.path.exists(get_path() + "pickled_datasets/classification.pickle"):
        class_datasets = load_datasets(
            get_path() + "pickled_datasets", "classification"
        )
    else:
        print(
            "Could not find preprocessed classification datasets, preprocessing them..."
        )
        class_datasets = get_datasets(classification_datasets)
        save_datasets(get_path() + "pickled_datasets", class_datasets, "classification")
    # regression
    if os.path.exists(get_path() + "pickled_datasets/regression.pickle"):
        reg_datasets = load_datasets(get_path() + "pickled_datasets", "regression")
    else:
        print("Could not find preprocessed regression datasets, preprocessing them...")
        reg_datasets = get_datasets(regression_datasets)
        save_datasets(get_path() + "pickled_datasets", reg_datasets, "regression")

    return class_datasets, reg_datasets


def main():
    random_state = 0
    print("loading datasets...")
    class_datasets, reg_datasets = get_pickled_datasets()
    # reg_datasets = [reg_datasets[0]]
    # print(reg_datasets[0])
    # encoding of categories
    # or, to get some single dataset:
    # class_datasets = get_datasets("default_credit")

    # Classification
    fit_and_save_models(
        class_datasets,
        classification_algos,
        random_state,
        "models",
        is_classification=True,
    )
    load_models_and_predict(
        class_datasets, classification_algos, random_state, "models"
    )
    # Regression
    # fit_and_save_models(
    #    reg_datasets, regression_algos, random_state, "models", is_classification=False
    # )
    # load_models_and_predict(
    #    reg_datasets, regression_algos, random_state, "models", is_classification=False
    # )


main()
