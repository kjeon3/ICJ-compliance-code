import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings

warnings.filterwarnings("ignore")

def find_best_combinations_statsmodels(data, target):
    if 'remedy_type' in data.columns:
        data['remedy_type'] = data['remedy_type'].astype('category').cat.codes

    predictors = [col for col in data.columns if col not in [target, 'compliance_binary']]
    individual_results = []

    for var in predictors:
        try:
            X = data[[var]]
            y = data[target]

            # Ordered logit model (logit link)
            model = OrderedModel(y, X, distr='logit')
            result = model.fit(method='bfgs', disp=False)

            coef = result.params[var]
            se = result.bse[var]
            p_val = result.pvalues[var]
            r2 = result.prsquared  # McFadden's pseudo-R²

            logit_eq = f"logit(P(Y ≤ j)) = θ_j - ({round(coef, 3)} × {var})"

            individual_results.append({
                "Variable": var,
                "Coefficient": round(coef, 3),
                "Standard Error": round(se, 4),
                "P-value": round(p_val, 4),
                "Pseudo R²": round(r2, 3),
                "Logit Equation": logit_eq
            })

        except Exception as e:
            print(f"⚠️ Skipping {var} due to error: {e}")

    return individual_results

# Your existing dataset
if __name__ == "__main__":
    data = pd.DataFrame({
        #compliance_level is the dependent variabe.
        # Ordinal. 0 is compliant, 1 is partial-compliance, 2 is non-compliance
        'compliance_level': [2, 0, 0, 0, 1, 1, 0, 0, 0, 2,
                             2, 2, 0, 0, 0, 0, 0, 0, 0, 2,
                             0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                             0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 2, 0, 1, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                             0, 1,],
        #from here these are my independent variables.
        'population_size': [1.25, 42.78, 43.23, 11.35, 1.78, 28.48, 77.98, 77.94, 570.48, 0.22,
                            0.22, 41.99, 3.72, 239.25, 25.67, 3.9, 0.35, 8.23, 8.53, 256.47,
                            5.27, 5.59, 4.31, 4.82, 1.78, 0.68, 0.7, 284.28, 222.09, 133.47,
                            10.33, 292.79, 8.43, 13.76, 7.48, 5.34, 7.48, 0.63, 62.78, 27.57,
                            4.81, 5.66, 3.32, 11.11, 13.31, 45.72, 73.25, 60.53, 17.7, 18.52,
                            69.85, 30.12, 17.86, 127.49, 4.82, 6.4, 5, 6.4, 6.4, 230.8,
                            17.27, 53.22,],

        'democracy_index': [0.09, 0.75, 0.75, 0.79, 0.17, 0.11, 0.8, 0.8, 0.61, 0.81,
                            0.82, 0.12, 0.07, 0.86, 0.84, 0.07, 0.66, 0.15, 0.19, 0.86,
                            0.49, 0.27, 0.88, 0.07, 0.67, 0.02, 0.11, 0.82, 0.67, 0.47,
                            0.89, 0.86, 0.63, 0.57, 0.52, 0.46, 0.63, 0.48, 0.89, 0.33,
                            0.39, 0.45, 0.91, 0.91, 0.69, 0.66, 0.33, 0.87, 0.6, 0.63,
                            0.43, 0.82, 0.89, 0.82, 0.91, 0.22, 0.89, 0.22, 0.22, 0.4,
                            0.17, 0.51],
        'Log_GDP_billions': [8, 10.6259, 10.642, 10.1644, 8.3802, 9.5211, 11.3979, 11.3979, 10.854, 9.1818,
                             9.1818, 11.0022, 10.5284, 12.6064, 11.5523, 10.4987, 9.0453, 9.3096, 9.2672, 12.7896,
                             9.6937, 9.7642, 11.0813, 10.4565, 9.5877, 10.244, 9.9533, 13.0253, 11.2915, 10.98,
                             11.4123, 13.0864, 9.8176, 9.6415, 10.0924, 9.8704, 10.6521, 9.5658, 12.4669, 11.3633,
                             11.287, 9.9191, 10.6227, 11.4521, 10.247, 11.569, 10.467, 12.3222, 10.1284, 10.0095,
                             11.6236, 11.3027, 11.4142, 12.6902, 10.7516, 10.1149, 10.7954, 10.1149, 10.1149, 11.5064,
                             9.9768, 11.0402,],
        'judge_from_country': [0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1],
        'military_strength': [0, 3.01, 3.28, 0.39, 0, 0.08, 5.15, 5.15, 2.5, 0,
                              0, 8.12, 0.71, 245, 7.35, 1, 0.0129, 0.017, 0.013, 299,
                              0.252, 0.134, 22.53, 0.55, 0.202, 2.77, 0.14, 0.33, 1.37, 0.58,
                              3.15, 493, 23.58, 17.3, 1.81, 0.72, 56.78, 0.046, 37.8, 14.72,
                              11.06, 0.848, 15.13, 5.13, 0.236, 11.7, 0.332, 29.8, 0.166, 0.106,
                              5.56, 3.22, 5.47, 46.9, 0, 0.0816, 0, 0.0816, 2.58, 10.22,
                              0.111, 1.19],
        'juris_type': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                       0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                       1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                       0, 0, 1, 1, 2, 2, 2, 2, 1, 1,
                       2, 2, 2, 2, 2, 2, 0, 2, 1, 1,
                       2, 2, 2, 0, 2, 2, 2, 2, 2, 2,
                       0, 0],
        'juris_challenge': [1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                            1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                            0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                            1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                            0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                            1, 0, 0, 1, 1, 0, 1, 1, 1, 1,
                            0, 1]
    })

    results = find_best_combinations_statsmodels(data, target="compliance_level")
    if results:
        results_df = pd.DataFrame(results).sort_values(by="P-value", na_position="last")
        print("\nOrdinal Regression Summary (statsmodels):")
        print(results_df.to_string(index=False))
    else:
        print("No valid models returned.")


def find_best_combinations_ordinal():
    return None
