# ==========================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# ==========================================================================

from pathlib import Path

# Folder locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_cell_growth_4factor.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

# Column names
# Written out by name: in this dataset the target is the FIRST column, so the
# df.columns[:-1] / df.columns[-1] shortcut would pick the wrong one.
TARGET_GROWTH_RATE = "Specific Growth Rate [d^-1]"
FEATURES = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]

# Settings
RANDOM_STATE = 0
TEST_SIZE = 0.2
FOLDS = 5

# Polynomial regression
POLYNOMIAL_DEGREES = [1, 2, 3, 4, 5, 6]  # 1 = straight-line model; higher = more curvature.

# k-nearest neighbours
KNN_NEIGHBORS = list(range(1, 41))     # k = 1, 2, ..., 40 neighbours to average.
KNN_WEIGHTS = ["uniform", "distance"]  # "uniform": all k neighbours count equally.
# "distance": closer neighbours count more.

# GridSearchCV
# make_pipeline() names each step after its class in lowercase, so the steps are
# "polynomialfeatures" and "kneighborsregressor". "step__setting" selects a setting of that step.
PARAM_GRID_POLY = dict(polynomialfeatures__degree=POLYNOMIAL_DEGREES)
PARAM_GRID_KNN = dict(kneighborsregressor__n_neighbors=KNN_NEIGHBORS,
                      kneighborsregressor__weights=KNN_WEIGHTS)

# Settings shared by every GridSearchCV
KWARGS_GSCV = dict(cv=FOLDS, scoring="neg_root_mean_squared_error", return_train_score=True)

# Learning curve: fractions of the full dataset to train on.
LEARNING_CURVE_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Plotting
FIG_WIDTH = 14.39 / 2.54   # Figure size in inches (converted from cm).
FIG_HEIGHT = 12.09 / 2.54
DPI = 200

FONTSIZE_XY_LABEL = 14
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 12

COLOR_POLY = "tab:blue"
COLOR_KNN = "tab:orange"

STR_DM1 = r"$d^{-1}$"  # Shows d with a proper superscript -1 in figures.