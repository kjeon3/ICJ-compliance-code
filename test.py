import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from mord import LogisticAT
from scipy import stats
import warnings

def approximate_p_value(model, X, y, B=100):
    coefs = []
    for _ in range(B):
        X_sample, y_sample = resample(X, y)
        try:
            m = LogisticAT()
            m.fit(X_sample, y_sample)
            coefs.append(m.coef_[0])
        except:
            continue
    if len(coefs) < 2:
        return None
    se = np.std(coefs)
    z = model.coef_[0] / se
    return 2 * (1 - stats.norm.cdf(np.abs(z)))

def find_best_combinations_ordinal(data, target):
    if 'remedy_type' in data.columns:
        data['remedy_type'] = data['remedy_type'].astype('category').cat.codes

    predictors = [col for col in data.columns if col not in [target, 'compliance_binary']]
    individual_results = []

    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    for var in predictors:
        try:
            X = data_scaled[[var]].values
            y = data_scaled[target].values

            model = LogisticAT()
            model.fit(X, y)
            y_pred = model.predict(X)
            from sklearn.metrics import log_loss

            y_proba = model.predict_proba(X)
            logloss_model = log_loss(y, y_proba)

            classes, counts = np.unique(y, return_counts=True)
            probs = counts / counts.sum()
            y_null_proba = np.tile(probs, (len(y), 1))
            logloss_null = log_loss(y, y_null_proba)

            pseudo_r2 = 1 - (logloss_model / logloss_null)

            r2 = np.corrcoef(y, y_pred)[0, 1] ** 2 if len(set(y_pred)) > 1 else 0

            coef = model.coef_[0]
            p_val = approximate_p_value(model, X, y)
            logit_eq = f"logit(P(Y ≤ j)) = θ_j - ({round(coef, 3)} × {var})"

            individual_results.append({
                "Variable": var,
                "Coefficient": round(coef, 3),
                "P-value": round(p_val, 4) if p_val else "N/A",
                "R²": round(r2, 3),
                "Pseudo R²": round(pseudo_r2, 3),
                "Logit Equation": logit_eq
            })

        except Exception as e:
            print(f"⚠️ Skipping {var} due to error: {e}")

    return individual_results

# Clumsy way of inputting data. Will fix later.
if __name__ == "__main__":
    data = pd.DataFrame({
        #compliance_level is the dependent variabe.
        # Ordinal. 0 is compliant, 1 is partial-compliance, 2 is non-compliance
        'compliance_level': [2, 0, 0, 0, 1, 1, 0, 0, 0, 2,
                             2, 1, 0, 0, 0, 0, 0, 0, 0, 2,
                             0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                             0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 2, 0, 1, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0, 0, 2,
                             0, 1, 1, 2],
        #from here these are my independent variables.
        'population_size': [1.25, 42.78, 43.23, 11.35, 1.78, 28.48, 77.98, 77.94, 570.48, 0.22,
                            0.22, 41.99, 3.72, 239.25, 25.67, 3.9, 0.35, 8.23, 8.53, 256.47,
                            5.27, 5.59, 4.31, 4.82, 1.78, 0.68, 0.7, 284.28, 222.09, 133.47,
                            10.33, 292.79, 8.43, 13.76, 7.48, 5.34, 7.48, 0.63, 62.78, 27.57,
                            4.81, 5.66, 3.32, 11.11, 13.31, 45.72, 73.25, 60.53, 17.7, 18.52,
                            69.85, 30.12, 17.86, 127.49, 4.82, 6.4, 5, 6.4, 6.4, 230.8,
                            17.27, 53.22, 47.31, 51.76],
        'remedy_type': ["Money", "Territory", "Sovereignty", "Sovereignty", "Behavior", "Military", "Behavior", "Behavior", "No Remedy", "Negotiate",
                        "Negotiate", "Behavior", "Behavior", "Maritime", "Territory", "Behavior", "Behavior", "Territory", "Territory", "Behavior",
                        "Territory", "Territory", "Territory", "Territory", "Sovereignty", "Behavior", "No Relief", "Behavior", "Sovereignty", "Sovereignty",
                        "Behavior", "Behavior", "Territory", "Territory", "Maritime", "Declaratory", "Declaratory", "Declaratory", "Declaratory", "Declaratory",
                        "Declaratory", "Behavior", "Declaratory", "Declaratory", "Behavior", "Maritime", "Money", "Declaratory", "Boundary", "Boundary",
                        "Territory", "Maritime", "Maritime", "Behavior", "Declaratory", "Declaratory", "Maritime", "Maritime", "Money", "Declaratory",
                        "Boundary", "Boundary", "Monetary", "Declaratory"],
        'democracy_index': [0.09, 0.75, 0.75, 0.79, 0.17, 0.11, 0.8, 0.8, 0.61, 0.81,
                            0.82, 0.12, 0.07, 0.86, 0.84, 0.07, 0.66, 0.15, 0.19, 0.86,
                            0.49, 0.27, 0.88, 0.07, 0.67, 0.02, 0.11, 0.82, 0.67, 0.47,
                            0.89, 0.86, 0.63, 0.57, 0.52, 0.46, 0.63, 0.48, 0.89, 0.33,
                            0.39, 0.45, 0.91, 0.91, 0.69, 0.66, 0.33, 0.87, 0.6, 0.63,
                            0.43, 0.82, 0.89, 0.82, 0.91, 0.22, 0.89, 0.22, 0.22, 0.4,
                            0.17, 0.51, 0.28, 0.7],
        'GDP_billions': [0.10, 42.26, 43.85, 14.6, 0.24, 3.32, 250, 250, 71.46, 1.52,
                         1.52, 100.5, 33.76, 4040, 356.72, 31.53, 1.11, 2.04, 1.85, 6160,
                         4.94, 5.81, 120.58, 28.61, 3.87, 17.54, 8.98, 10600, 195.66, 95.5,
                         258.38, 12200, 6.57, 4.38, 12.37, 7.42, 44.89, 3.68, 2930, 230.81,
                         193.62, 8.3, 41.95, 283.23, 17.66, 370.69, 29.31, 2100, 13.44, 10.22,
                         420.33, 200.79, 259.56, 4900, 56.44, 13.03, 62.42, 13.03, 13.03, 320.91,
                         9.48, 109.7, 45.57, 345.33],
        'judge_from_country': [0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1],
        'military_strength': [0, 3.01, 3.28, 0.39, 0, 0.08, 5.15, 5.15, 2.5, 0,
                              0, 8.12, 0.71, 245, 7.35, 1, 0.0129, 0.017, 0.013, 299,
                              0.252, 0.134, 22.53, 0.55, 0.202, 2.77, 0.14, 0.33, 1.37, 0.58,
                              3.15, 493, 23.58, 17.3, 1.81, 0.72, 56.78, 0.046, 37.8, 14.72,
                              11.06, 0.848, 15.13, 5.13, 0.236, 11.7, 0.332, 29.8, 0.166, 0.106,
                              5.56, 3.22, 5.47, 46.9, 0, 0.0816, 0, 0.0816, 2.58, 10.22,
                              0.111, 1.19, 1.07, 9.66],
        'juris_type': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                       0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                       1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                       0, 0, 1, 1, 2, 2, 2, 2, 1, 1,
                       2, 2, 2, 2, 2, 2, 0, 2, 1, 1,
                       2, 2, 2, 0, 2, 2, 2, 2, 2, 2,
                       0, 0, 0, 2],
        'juris_challenge': [1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                            1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                            0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                            1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                            0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                            1, 0, 0, 1, 1, 0, 1, 1, 1, 1,
                            0, 1, 1, 1]
    })

    results = find_best_combinations_ordinal(data, target="compliance_level")
    if results:
        results_df = pd.DataFrame(results).sort_values(by="P-value", na_position="last")
        print("\n Ordinal Regression Summary:")
        print(results_df.to_string(index=False))
    else:
        print("No valid models returned.")
