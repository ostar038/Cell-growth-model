
#   Predicts the specific growth rate of cells from temperature, pH,
#     glucose concentration, and lactate concentration.
#
#   Polynomial regression and k-nearest neighbors (kNN) regression are
#     each tuned with GridSearchCV, compared on a held-out test set, and
#     then used to explore how each variable affects growth rate.
#
#   main.py only calls code from src/. Each numbered block below is one
#     stage of the workflow.


import os

import pandas as pd

from src.classes import ModelCandidate
from src.constants import (
    COLOR_KNN, COLOR_POLY, FIGURES_DIR, PARAM_GRID_KNN, PARAM_GRID_POLY, TABLES_DIR,
)
from src.functions import (
    compute_learning_curve, get_x_y, load_data, make_knn_pipeline, make_polynomial_pipeline,
    plot_learning_curve, plot_single_factor_effects,
    plot_validation_curve_knn, plot_validation_curve_poly, save_best_and_worst_conditions,
    save_polynomial_weights, save_summary_table, split_data,
)


def main():
    # Make sure the output folders exist (no error if they already do).
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # ==========================
    #   1. Load and explore
    # ==========================
    df = load_data()
    print(f"\nLoaded {df.shape[0]} samples with {df.shape[1]} columns.")
    save_summary_table(df)

    # ==========================
    #   2. Split into training and test sets
    # ==========================
    x, y = get_x_y(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    print(f"Training samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}\n")

    # ==========================
    #   3. Tune both models with GridSearchCV (training set only)
    # ==========================
    poly = ModelCandidate("Poly", make_polynomial_pipeline(), PARAM_GRID_POLY, COLOR_POLY)
    knn = ModelCandidate("kNN", make_knn_pipeline(), PARAM_GRID_KNN, COLOR_KNN)
    candidates = [poly, knn]

    for i, candidate in enumerate(candidates, start=2):
        print(f"Tuning {candidate.label}...")
        candidate.tune(x_train, y_train)
        candidate.check_grid_edges()
        candidate.results_table().to_csv(
            os.path.join(TABLES_DIR, f"table_0{i}_cv_results_{candidate.label.lower()}.csv"), index=False)

    plot_validation_curve_poly(poly.results_table())
    plot_validation_curve_knn(knn.results_table())

    # ==========================
    #   4. Compare both models on the held-out test set
    # ==========================
    df_comparison = pd.DataFrame([candidate.evaluate(x_train, y_train, x_test, y_test) for candidate in candidates])
    df_comparison.to_csv(os.path.join(TABLES_DIR, "table_04_model_comparison.csv"), index=False)

    print("\nModel comparison:\n")
    print(df_comparison.round(5).to_string(index=False))

    # Model selection: if two models have similar scores, prefer the simpler,
    # more interpretable one. "Similar" here means the difference in CV RMSE is smaller than the
    # fold-to-fold standard deviation of the lowest-scoring model, i.e., within the CV noise.
    # Selection uses CV RMSE, not test RMSE, so the test set stays an unbiased estimate.
    # The candidates list is ordered from most to least interpretable (Poly first).
    lowest = min(candidates, key=lambda candidate: candidate.cv_rmse)
    tolerance = lowest.cv_rmse_std
    best = next(c for c in candidates if c.cv_rmse <= lowest.cv_rmse + tolerance)

    print(f"\nLowest CV RMSE: {lowest.name} ({lowest.cv_rmse:.4f} +/- {tolerance:.4f} {'d^-1'})")
    if best is not lowest:
        print(f"{best.name} is within one standard deviation ({best.cv_rmse:.4f} d^-1), "
              f"so the more interpretable model is preferred.")
    print(f"Selected model: {best.name}")

    # ==========================
    #   5. Interpret the selected model
    # ==========================
    plot_single_factor_effects(candidates, df)
    save_best_and_worst_conditions(best, df)

    # Polynomial weights: interpret how each feature affects growth rate.
    df_weights = save_polynomial_weights(poly)
    print(f"\nLargest polynomial weights ({poly.name}, standardized features):\n")
    print(df_weights.head(11).round(4).to_string(index=False))

    # ==========================
    #   6. Learning curve
    # ==========================
    print("\nComputing learning curve...")
    df_lc = compute_learning_curve(x, y)
    df_lc.to_csv(os.path.join(TABLES_DIR, "table_07_learning_curve.csv"), index=False)
    plot_learning_curve(df_lc)

    print(f"\nFigures saved to '{FIGURES_DIR}' and tables saved to '{TABLES_DIR}'.\n")


if __name__ == "__main__":
    main()