import scipy.stats
import sklearn.svm
import sklearn.linear_model
import sklearn.tree
import sklearn.neural_network
import sklearn.gaussian_process
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.model_selection import RandomizedSearchCV


def mlpr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    mlp = sklearn.neural_network.MLPRegressor()
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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def dtr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    dt = sklearn.tree.DecisionTreeRegressor()

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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def rfr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    rf = RandomForestRegressor()

    param_distributions = {
        "max_depth": scipy.stats.reciprocal(10, 3000),
        "n_estimators": scipy.stats.reciprocal(100, 2000),
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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def adaboostr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    abc = AdaBoostRegressor()
    param_distributions = {
        "n_estimators": scipy.stats.reciprocal(50, 1000),
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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def gboostr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    gbc = GradientBoostingRegressor()
    param_distributions = {
        "max_depth": [i for i in range(1, 11)],
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
    print(search.cv_results_)
    print("best parameters:", search.best_params_)
    print(
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def svr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    svm = sklearn.svm.SVR(kernel="rbf")
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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def linr(X, y, random_state=0, scoring="r2", n_iter=4, is_multilabel=None):
    lr = sklearn.linear_model.LinearRegression()
    param_distributions = {
        "fit_intercept": [True, False],
        "normalize": [True, False],
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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_


def gpr(X, y, random_state=0, scoring="r2", n_iter=30, is_multilabel=None):

    gp = sklearn.gaussian_process.GaussianProcessRegressor()
    param_distributions = {
        "normalize_y": [True, False],
        "alpha": scipy.stats.reciprocal(10e-10, 5.0),
    }
    search = RandomizedSearchCV(
        gp,
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
        "%.1f%% R2 score on validation sets (average)" % (search.best_score_ * 100),
        "\n",
    )
    return search.best_estimator_
