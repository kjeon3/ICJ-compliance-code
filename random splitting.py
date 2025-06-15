import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler

from pvalues import approximate_p_value


def find_best_combinations_ordinal(data, target, training_sample_ratio=0.5):
    if 'remedy_type' in data.columns:
        data['remedy_type'] = data['remedy_type'].astype('category').cat.codes

    predictors = [col for col in data.columns if col not in [target, 'compliance_binary']]
    individual_results = []

    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[predictors] = scaler.fit_transform(data[predictors])

    cnt_training_samples = int(data.shape[0] * training_sample_ratio)

    for var in predictors:
        try:
            X = data_scaled[[var]].values
            y = data_scaled[target].values

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=training_sample_ratio, random_state=42
            )

            print(f"✅ Split {len(X_train)} training / {len(X_test)} testing rows for {var}")

            model = LogisticAT()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            from sklearn.metrics import log_loss
            logloss_model = log_loss(y_test, y_proba)

            # Null model using class distribution
            classes, counts = np.unique(y_test, return_counts=True)
            probs = counts / counts.sum()
            y_null_proba = np.tile(probs, (len(y_test), 1))
            logloss_null = log_loss(y_test, y_null_proba)

            pseudo_r2 = 1 - (logloss_model / logloss_null)
            r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2 if len(set(y_pred)) > 1 else 0

            coef = model.coef_[0]
            p_val = approximate_p_value(model, X_test, y_test)

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
        'log_GDP_Diff': [0.4437, 2.0612, 2.9583, 0.0922, 1.7139, 0.7016, 0.8657, 1.2101, 0.8799, 2.4684,
                         2.1323, 1.5043, 0.6183, 1.0541, 1.0541, 1.4534, 1.4534, 0.0424, 0.0424, 3.6164,
                         0.0705, 0.0705, 0.0744, 1.3846, 0.1511, 0.2907, 0.2907, 0.7308, 0.2878, 0.8859,
                         1.4713, 1.1729, 0.1761, 0.1761, 0.222, 0.222, 0.454, 0.6323, 3.4673, 0.0763,
                         0.0763, 0.5687, 1.0043, 1.4313, 1.4506, 1.5466, 0.5839, 0.2341, 0.1189, 0.1189,
                         1.3267, 0.1115, 0.1115, 0.5229, 0.6366, 0.6804, 0.6805, 0.68051, 0.6804, 0.9469,
                         1.0634, 1.0634, 0.159, 1.3437],
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

