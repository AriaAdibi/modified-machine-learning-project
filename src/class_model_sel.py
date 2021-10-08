from math import sqrt
import scipy.stats
import sklearn.svm
import sklearn.linear_model
import sklearn.tree
import sklearn.neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import RandomizedSearchCV
from utils import is_imbalanced


def mlpc(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    mlp = sklearn.neural_network.MLPClassifier()
    param_distributions = {
        "hidden_layer_sizes": [(50,), (100,), (200,), (100, 50)],
        "activation": ["relu", "tanh", "logistic"],
        "alpha": scipy.stats.reciprocal(0.00001, 0.1),
    }
    search = RandomizedSearchCV(
        mlp,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def dtc(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    if is_imbalanced(y):
        dt = sklearn.tree.DecisionTreeClassifier(class_weight="balanced")
    else:
        dt = sklearn.tree.DecisionTreeClassifier()

    param_distributions = {
        "max_depth": scipy.stats.reciprocal(10, 3000),
    }
    search = RandomizedSearchCV(
        dt,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def rfc(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    if is_imbalanced(y):
        rf = RandomForestClassifier(class_weight="balanced_subsample")
    else:
        rf = RandomForestClassifier()

    param_distributions = {
        "max_depth": scipy.stats.reciprocal(10, 3000),
        "n_estimators": list(range(100, 2000)),
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def adaboostc(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    abc = AdaBoostClassifier()
    param_distributions = {
        "n_estimators": list(range(1, 100)),
        "learning_rate": scipy.stats.reciprocal(0.01, 10.0),
    }
    search = RandomizedSearchCV(
        abc,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def gboostc(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    gbc = GradientBoostingClassifier()
    param_distributions = {
        "max_depth": list(range(1, 11)),
        "learning_rate": scipy.stats.reciprocal(0.01, 10.0),
    }
    search = RandomizedSearchCV(
        gbc,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)

    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def svc(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):
    # handle multilabel as a pipeline
    if is_multilabel:
        print("SVM MULTILABEL")
        # return svm_multilabel(X, y, random_state=0, scoring="f1", n_iter=30)

    if is_imbalanced(y):
        svm = sklearn.svm.SVC(kernel="rbf", class_weight="balanced")
    else:
        svm = sklearn.svm.SVC(kernel="rbf")
    param_distributions = {
        "C": scipy.stats.reciprocal(1.0, 1000.0),
        "gamma": scipy.stats.reciprocal(0.01, 10.0),
    }
    search = RandomizedSearchCV(
        svm,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def gnb(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    gnb = GaussianNB()
    param_distributions = {
        "var_smoothing": scipy.stats.reciprocal(10e-10, 10e-4),
    }
    search = RandomizedSearchCV(
        gnb,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def logr(X, y, random_state=0, scoring="f1", n_iter=30, is_multilabel=None):

    if is_imbalanced(y):
        lr = sklearn.linear_model.LogisticRegression(
            solver="lbfgs", class_weight="balanced"
        )
    else:
        lr = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    param_distributions = {
        "C": scipy.stats.reciprocal(0.001, 100.0),
        "tol": [0.0001, 0.00001, 0.000001],
    }
    search = RandomizedSearchCV(
        lr,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        random_state=random_state,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def knn(X, y, random_state=0, scoring="f1", n_iter=15, is_multilabel=False):
    # returns a fitted KNeighborsClassifier
    # Tunes k based on randomised search, from a range centered at sqrt(n)
    # we chose Stratified KFold, because we wanted:
    # 1. to always split in the same way for all datasets
    # 2. To properly deal with unbalanced datasets
    # 3. despite the fact that stratification can result in
    #       misrepresentation when our dataset is imbalanced and there is a
    #       number of difficult overlap on the features, which can result
    #       in problematic splits, that we are willing to sacrifice this
    #       in favour of keeping the same parameters across the
    #       different datasets.

    knn = KNeighborsClassifier()

    if is_imbalanced(y):
        r = [1, 3, 5]
        n_iter = 3
    else:
        # justification of sqrt(n) -> pattern classification by duda
        sq = int(sqrt(len(X)))
        r = range(int(sq - sq / 2), int(sq + sq / 2))

    params = dict(n_neighbors=r)
    search = RandomizedSearchCV(
        knn,
        params,
        cv=3,
        n_jobs=-1,
        n_iter=n_iter,
        scoring=scoring,
        random_state=random_state,
        refit=True,
    )
    search.fit(X, y)
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% accuracy on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def svm_multilabel(X, y, random_state=0, scoring="f1", n_iter=30):
    return None
