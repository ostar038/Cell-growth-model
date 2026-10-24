
#   * Every stage of the workflow is one function. main.py calls them in order:
#       1. Load and explore     load_data, save_summary_table
#       2. Split                get_x_y, split_data
#       3. Build models         make_polynomial_pipeline, make_knn_pipeline
#       4. Validation curves    plot_validation_curve_poly, plot_validation_curve_knn
#       5. Interpret            plot_single_factor_effects, save_best_and_worst_conditions,
#                               save_polynomial_weights
#       6. Learning curve       compute_learning_curve, plot_learning_curve
#
#   * Figures produced: validation
#     curves, model output vs one feature with the others held
#     at their mean, and learning curves.
#
#   * Tuning and evaluation live in the ModelCandidate class (classes.py).
#


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.classes import ModelCandidate
from src.constants import (
    COLOR_KNN, COLOR_POLY, DATASET_PATH, DPI, FEATURES, FIG_HEIGHT, FIG_WIDTH, FIGURES_DIR,
    FONTSIZE_LEGEND, FONTSIZE_TICKS, FONTSIZE_XY_LABEL, LEARNING_CURVE_FRACTIONS,
    PARAM_GRID_KNN, PARAM_GRID_POLY, RANDOM_STATE, STR_DM1, TABLES_DIR, TARGET_GROWTH_RATE, TEST_SIZE,
)

# By default, matplotlib makes math in italics, but we want our units in regular.
plt.rcParams['mathtext.default'] = 'regular'


# ==========================
#   Small plotting helpers
# ==========================

def axis_label(name):
    """Replace 'd^-1' in a column name with a properly formatted superscript."""
    return name.replace("d^-1", STR_DM1)


def style_axes(ax, xlabel, ylabel):
    """Apply consistent labels and font sizes."""
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_XY_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_XY_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)


def save_figure(fig, filename):
    fig.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close(fig)


# ==========================
#   1. Load and explore
# ==========================

def load_data():
    """Import the dataset and keep only complete rows of the five columns we need."""
    df = pd.read_csv(DATASET_PATH)
    df = df.loc[:, FEATURES + [TARGET_GROWTH_RATE]]
    df = df.dropna()  # Remove any row with a missing value.
    return df


def save_summary_table(df):
    """Min, max, mean, etc. of each column. The min/max define the range the model has seen."""
    summary = df.describe().T  # .T puts one variable per row.
    summary.to_csv(os.path.join(TABLES_DIR, "table_01_dataset_summary.csv"))


# ==========================
#   2. Split
# ==========================

def get_x_y(df):
    """Convert the feature and target columns to numpy arrays."""
    x = df.loc[:, FEATURES].to_numpy()
    y = df.loc[:, TARGET_GROWTH_RATE].to_numpy()
    return x, y


def split_data(x, y):
    """Split the data into training and testing sets."""
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ==========================
#   3. Build the model pipelines
# ==========================

def make_polynomial_pipeline():
    """Standardize -> polynomial features -> linear regression."""
    # "degree" is left undefined here because GridSearchCV will try different values.
    return make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())


def make_knn_pipeline():
    """Standardize -> kNN regression. Scaling matters because kNN uses distances."""
    return make_pipeline(StandardScaler(), KNeighborsRegressor())


# ==========================
#   4. Validation curves
# ==========================

def plot_validation_curve_poly(table):
    """Training and validation RMSE vs polynomial degree."""
    table = table.sort_values(by=["param_polynomialfeatures__degree"], ascending=True)
    degrees = table.loc[:, "param_polynomialfeatures__degree"].to_numpy()
    rmse_train = table.loc[:, "rmse_train"].to_numpy()
    rmse_valid = table.loc[:, "rmse_valid"].to_numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    ax.plot(degrees, rmse_train, label="Train (Poly)", color=COLOR_POLY, linestyle="-", marker="o")
    ax.plot(degrees, rmse_valid, label="Valid (Poly)", color=COLOR_POLY, linestyle="--", marker="o")
    style_axes(ax, "Polynomial Degree", f"Specific Growth Rate RMSE [{STR_DM1}]")
    ax.legend(ncols=1, fontsize=FONTSIZE_LEGEND)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # High degrees can overfit so badly that their validation RMSE dwarfs everything else.
    # Cap the y-axis at 3x the best validation RMSE so the region of interest stays readable.
    ax.set_ylim(bottom=0, top=3 * np.min(rmse_valid))

    save_figure(fig, "figure_01_validation_curve_poly.png")


def plot_validation_curve_knn(table):
    """Training and validation RMSE vs k, one pair of lines per weighting."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    colors = ["tab:red", "tab:purple"]

    for color, weight in zip(colors, PARAM_GRID_KNN["kneighborsregressor__weights"]):
        mask_weight = (table.loc[:, "param_kneighborsregressor__weights"] == weight)
        df_sub = table.loc[mask_weight, :].sort_values(by=["param_kneighborsregressor__n_neighbors"])
        n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()

        ax.plot(n_neighbors, df_sub.loc[:, "rmse_train"], label=f"Train (kNN: {weight})", color=color, linestyle="-")
        ax.plot(n_neighbors, df_sub.loc[:, "rmse_valid"], label=f"Valid (kNN: {weight})", color=color, linestyle="--")

    style_axes(ax, "k", f"Specific Growth Rate RMSE [{STR_DM1}]")
    ax.legend(ncols=1, fontsize=FONTSIZE_LEGEND)
    ax.set_ylim(bottom=0)

    save_figure(fig, "figure_02_validation_curve_knn.png")


# ==========================
#   5. Interpret the models
# ==========================

def plot_single_factor_effects(candidates, df, num=200):
    """Vary one feature across its observed range while the others are held at their mean."""
    x_all = df.loc[:, FEATURES].to_numpy()
    means = np.mean(x_all, axis=0)  # Mean of each column.

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(2 * FIG_WIDTH, 2 * FIG_HEIGHT), dpi=DPI,
                            layout="constrained", sharey=True)

    for i, (ax, feature) in enumerate(zip(axs.flat, FEATURES)):
        # num rows, every column at its mean...
        x_plot = np.tile(means, (num, 1))
        # ...then overwrite column i with evenly spaced values across its range.
        x_plot[:, i] = np.linspace(np.min(x_all[:, i]), np.max(x_all[:, i]), num=num)

        linestyles = [":", "--"]
        for candidate, linestyle in zip(candidates, linestyles):
            ax.plot(x_plot[:, i], candidate.predict(x_plot), label=candidate.name, color=candidate.color,
                    linestyle=linestyle, linewidth=2)
        style_axes(ax, feature, axis_label(TARGET_GROWTH_RATE))

    axs.flat[0].legend(loc="upper left", fontsize=FONTSIZE_LEGEND - 2)
    save_figure(fig, "figure_03_single_factor_effects.png")


def save_best_and_worst_conditions(candidate, df, num=12):
    """Predict on a grid of conditions; save the 10 highest and 10 lowest predicted growth rates.

    The grid spans the 5th to 95th percentile of each feature rather than the full min-max range.
    The corners of the full range (e.g., 30 C AND pH 6.0 at the same time) contain almost no
    samples, so the polynomial extrapolates there and can predict impossible negative growth
    rates. Restricting the grid keeps the search where the data actually is.
    """
    x_all = df.loc[:, FEATURES].to_numpy()
    lows = np.percentile(x_all, 5, axis=0)    # 5th percentile of each column.
    highs = np.percentile(x_all, 95, axis=0)  # 95th percentile of each column.
    axes_values = [np.linspace(lows[i], highs[i], num=num) for i in range(len(FEATURES))]

    # meshgrid with 4 inputs gives every combination (num^4 rows once flattened).
    grids = np.meshgrid(*axes_values, indexing="ij")
    x_grid = np.column_stack([grid.ravel() for grid in grids])

    df_grid = pd.DataFrame(x_grid, columns=FEATURES)
    df_grid[f"Predicted {TARGET_GROWTH_RATE}"] = candidate.predict(x_grid)
    df_grid = df_grid.sort_values(by=[f"Predicted {TARGET_GROWTH_RATE}"], ascending=False)

    df_grid.head(10).to_csv(os.path.join(TABLES_DIR, "table_05_conditions_high_growth.csv"), index=False)
    df_grid.tail(10).to_csv(os.path.join(TABLES_DIR, "table_06_conditions_low_growth.csv"), index=False)


def save_polynomial_weights(candidate):
    """Write out the polynomial as readable terms with their weights.

    Because the features were standardized before PolynomialFeatures, the weights are on the
    same scale and can be compared: a large weight means a strong effect. A large negative
    squared term (e.g., [Temperature]^2) implies an optimum for that feature.
    """
    # The best pipeline found by GridSearchCV. Steps can be accessed by name.
    pipeline = candidate.model_gscv.best_estimator_
    powers = pipeline['polynomialfeatures'].powers_      # Power of each original feature in each term.
    coefs = pipeline['linearregression'].coef_           # Weight of each term.
    intercept = pipeline['linearregression'].intercept_  # Baseline growth rate.

    # Since we are using standardized features, we remove the units from the names.
    feature_names = [feature.split(" [")[0] for feature in FEATURES]

    terms = []
    for i in range(powers.shape[0]):
        partial_terms = []
        for j in range(powers.shape[1]):
            if powers[i, j] == 0:
                pass  # The j-th feature is not used in the i-th term.
            elif powers[i, j] == 1:
                partial_terms.append(f"[{feature_names[j]}]")
            else:
                partial_terms.append(f"[{feature_names[j]}]^{powers[i, j]}")
        terms.append(" * ".join(partial_terms))

    df_weights = pd.DataFrame({"Term": terms, "Weight": coefs})
    df_weights["Absolute Weight"] = np.abs(df_weights["Weight"])
    df_weights = df_weights.sort_values(by=["Absolute Weight"], ascending=False)

    # Put the intercept (baseline) on the first row.
    df_intercept = pd.DataFrame({"Term": ["Intercept (baseline)"], "Weight": [intercept],
                                 "Absolute Weight": [np.abs(intercept)]})
    df_weights = pd.concat([df_intercept, df_weights], ignore_index=True)

    df_weights.to_csv(os.path.join(TABLES_DIR, "table_08_polynomial_weights.csv"), index=False)
    return df_weights


# ==========================
#   6. Learning curve
# ==========================

def compute_learning_curve(x_all, y_all):
    """Re-tune and evaluate both models on progressively larger random subsets of the data."""
    rng = np.random.default_rng(seed=RANDOM_STATE)
    rows = []

    for fraction in LEARNING_CURVE_FRACTIONS:
        n_samples = round(fraction * x_all.shape[0])

        # Select a random subset of n_samples from the dataset.
        idxs = rng.choice(x_all.shape[0], size=n_samples, replace=False)
        x_train, x_test, y_train, y_test = split_data(x_all[idxs], y_all[idxs])

        row = {"n_training_samples": x_train.shape[0]}
        candidates = [ModelCandidate("Poly", make_polynomial_pipeline(), PARAM_GRID_POLY, COLOR_POLY),
                      ModelCandidate("kNN", make_knn_pipeline(), PARAM_GRID_KNN, COLOR_KNN)]
        for candidate in candidates:
            candidate.tune(x_train, y_train)
            row[f"rmse_train_{candidate.label}"] = root_mean_squared_error(y_train, candidate.predict(x_train))
            row[f"rmse_test_{candidate.label}"] = root_mean_squared_error(y_test, candidate.predict(x_test))
            row[f"best_{candidate.label}"] = candidate.name
        rows.append(row)

    return pd.DataFrame(rows)


def plot_learning_curve(df_lc):
    """Training and test RMSE vs number of training samples, for both models."""
    n = df_lc.loc[:, "n_training_samples"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    ax.plot(n, df_lc.loc[:, "rmse_train_Poly"], label="RMSE Train (Poly)", color=COLOR_POLY, linestyle="-")
    ax.plot(n, df_lc.loc[:, "rmse_test_Poly"], label="RMSE Test (Poly)", color=COLOR_POLY, linestyle="--")
    ax.plot(n, df_lc.loc[:, "rmse_train_kNN"], label="RMSE Train (kNN)", color=COLOR_KNN, linestyle="-")
    ax.plot(n, df_lc.loc[:, "rmse_test_kNN"], label="RMSE Test (kNN)", color=COLOR_KNN, linestyle="--")
    style_axes(ax, "Number of training samples", f"RMSE [{STR_DM1}]")
    ax.legend(ncols=1, fontsize=FONTSIZE_LEGEND)
    ax.set_ylim(bottom=0)

    save_figure(fig, "figure_04_learning_curve.png")