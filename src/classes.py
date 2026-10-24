# ============================================================================
#   CLASSES
# ============================================================================
#
#   * Both models (polynomial and kNN) go through exactly the same steps:
#       - tune the hyperparameters with GridSearchCV
#       - report every combination that was tried
#       - evaluate the best model on the training and test sets
#
#   * ModelCandidate writes those steps once. Each model becomes one
#     ModelCandidate object carrying its own pipeline, grid, and results.
#
# ============================================================================

import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.constants import KWARGS_GSCV


class ModelCandidate:
    """One model: its pipeline, the hyperparameters to search, and its results."""

    def __init__(self, label, model, param_grid, color):
        # __init__ runs automatically when the object is created.
        # self.something = ... stores a value on the object (an "attribute").
        self.label = label            # Short name used in legends, e.g. "Poly" or "kNN".
        self.model = model            # The untrained pipeline from make_pipeline().
        self.param_grid = param_grid  # The "Grid" to "Search" in GridSearchCV.
        self.color = color            # Plot colour used for this model in every figure.

        # Filled in by tune(). None means "not tuned yet".
        self.model_gscv = None
        self.best_params = None
        self.cv_rmse = None
        self.cv_rmse_std = None
        self.name = label

    def tune(self, x_train, y_train):
        """Run GridSearchCV on the training set and store the best hyperparameters."""
        self.model_gscv = GridSearchCV(self.model, param_grid=self.param_grid, **KWARGS_GSCV)
        self.model_gscv.fit(x_train, y_train)

        self.best_params = self.model_gscv.best_params_
        # Recall that we used negative RMSE, so flip the sign to get a positive RMSE.
        self.cv_rmse = -self.model_gscv.best_score_
        # Standard deviation of the validation RMSE across the CV folds, for the best hyperparameters.
        self.cv_rmse_std = self.model_gscv.cv_results_["std_test_score"][self.model_gscv.best_index_]

        # A readable name that includes the chosen hyperparameters,
        # e.g. "Poly (degree=3)". key.split("__")[-1] turns
        # "polynomialfeatures__degree" into "degree".
        settings = ", ".join(f"{key.split('__')[-1]}={value}" for key, value in self.best_params.items())
        self.name = f"{self.label} ({settings})"

    def check_grid_edges(self):
        """Warn if a best hyperparameter is the largest value searched.

        If so, an even larger value might have been better, so the range should be extended.
        """
        for key, values in self.param_grid.items():
            best = self.best_params[key]
            is_numeric = all(isinstance(value, (int, float)) for value in values)
            if is_numeric and best == max(values):
                print(f"  Note: best {key.split('__')[-1]}={best} is the largest value searched. "
                      f"Consider extending the range in constants.py.")

    def predict(self, x):
        """Predict with the best model. GridSearchCV refits it on the whole training set."""
        return self.model_gscv.predict(x)

    def results_table(self):
        """Every hyperparameter combination tried, sorted best to worst."""
        df = pd.DataFrame(self.model_gscv.cv_results_)

        # Sort by rank_test_score (i.e., best to worst).
        df = df.sort_values(by=["rank_test_score"], ascending=True)

        # Columns of interest: the rank, the hyperparameters, and the scores.
        param_columns = [f"param_{key}" for key in self.param_grid]
        columns_view = ["rank_test_score"] + param_columns + ["mean_test_score", "std_test_score",
                                                              "mean_train_score", "std_train_score"]
        table = df.loc[:, columns_view].copy()

        # Add positive RMSE columns, which are easier to read in a report.
        # (+ 0.0 turns a "negative zero" into a plain 0.0 when the training error is exactly zero.)
        table["rmse_valid"] = -table["mean_test_score"] + 0.0
        table["rmse_train"] = -table["mean_train_score"] + 0.0
        return table

    def evaluate(self, x_train, y_train, x_test, y_test):
        """One row of the model comparison table."""
        y_train_pred = self.predict(x_train)
        y_test_pred = self.predict(x_test)
        return {
            "Model": self.name,
            "CV RMSE [d^-1]": self.cv_rmse,
            "CV RMSE SD [d^-1]": self.cv_rmse_std,
            "Train RMSE [d^-1]": root_mean_squared_error(y_train, y_train_pred),
            "Test RMSE [d^-1]": root_mean_squared_error(y_test, y_test_pred),
            "Train R^2": r2_score(y_train, y_train_pred),
            "Test R^2": r2_score(y_test, y_test_pred),
        }
