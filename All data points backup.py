import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mord import LogisticAT
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from scipy import stats

from pvalues import find_best_combinations_ordinal


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
    se = np.std(coefs, axis=0)
    z = model.coef_[0] / se
    return 2 * (1 - stats.norm.cdf(np.abs(z)))

def run_full_ordinal_model(data, target="compliance_level", training_sample_ratio=0.7):
    # Encode categorical
    if 'remedy_type' in data.columns:
        data['remedy_type'] = data['remedy_type'].astype('category').cat.codes

    predictors = [
        'population_size', 'democracy_index', 'GDP_billions',
        'judge_from_country', 'military_strength', 'juris_challenge',
        'remedy_type', 'juris_type'
    ]

    # Scale predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[predictors])
    y = data[target].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=training_sample_ratio, random_state=42
    )

    model = LogisticAT()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Pseudo R²
    from sklearn.metrics import log_loss
    logloss_model = log_loss(y_test, y_proba)
    probs = np.bincount(y_test) / len(y_test)
    y_null = np.tile(probs, (len(y_test), 1))
    logloss_null = log_loss(y_test, y_null)
    pseudo_r2 = 1 - logloss_model / logloss_null

    # R²
    r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2 if len(set(y_pred)) > 1 else 0

    # Coefficients and p-values
    coefs = model.coef_[0]
    p_vals = approximate_p_value(model, X_test, y_test)

    print("\n📊 Multivariate Ordinal Regression Results:")
    print(f"Pseudo R²: {round(pseudo_r2, 3)}")
    print(f"R²: {round(r2, 3)}\n")

    print(f"{'Variable':<20}{'Coefficient':>12}{'P-value':>12}")
    print("-" * 44)
    for i, var in enumerate(predictors):
        pval = round(p_vals[i], 4) if p_vals is not None else "N/A"
        print(f"{var:<20}{round(coefs[i], 3):>12}{pval:>12}")

    # Output full logit equation
    terms = " + ".join([f"({round(coefs[i], 3)} × {predictors[i]})" for i in range(len(predictors))])
    print("\nLogit Equation:\nlogit(P(Y ≤ j)) = θ_j - [" + terms + "]")

# Run it
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
        #binary if ever needed. 0 is compliant, 1 is noncompliant.
        #[1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
        # 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        # 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
        # 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         #0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
         #0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        #0, 1, 1, 1],
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
        'Log_GDP_billions': [8, 10.6259, 10.642, 10.1644, 8.3802, 9.5211, 11.3979, 11.3979, 10.854, 9.1818,
                             9.1818, 11.0022, 10.5284, 12.6064, 11.5523, 10.4987, 9.0453, 9.3096, 9.2672, 12.7896,
                             9.6937, 9.7642, 11.0813, 10.4565, 9.5877, 10.244, 9.9533, 13.0253, 11.2915, 10.98,
                             11.4123, 13.0864, 9.8176, 9.6415, 10.0924, 9.8704, 10.6521, 9.5658, 12.4669, 11.3633,
                             11.287, 9.9191, 10.6227, 11.4521, 10.247, 11.569, 10.467, 12.3222, 10.1284, 10.0095,
                             11.6236, 11.3027, 11.4142, 12.6902, 10.7516, 10.1149, 10.7954, 10.1149, 10.1149, 11.5064,
                             9.9768, 11.0402, 10.6587, 11.5382],
        'GDP_billions': [8, 10.6259, 10.642, 10.1644, 8.3802, 9.5211, 11.3979, 11.3979, 10.854, 9.1818,
                         9.1818, 11.0022, 10.5284, 12.6064, 11.5523, 10.4987, 9.0453, 9.3096, 9.2672, 12.7896,
                         9.6937, 9.7642, 11.0813, 10.4565, 9.5877, 10.244, 9.9533, 13.0253, 11.2915, 10.98,
                         11.4123, 13.0864, 9.8176, 9.6415, 10.0924, 9.8704, 10.6521, 9.5658, 12.4669, 11.3633,
                         11.287, 9.9191, 10.6227, 11.4521, 10.247, 11.569, 10.467, 12.3222, 10.1284, 10.0095,
                         11.6236, 11.3027, 11.4142, 12.6902, 10.7516, 10.1149, 10.7954, 10.1149, 10.1149, 11.5064,
                         9.9768, 11.0402, 10.6587, 11.5382],
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
        #this is with 0 compulsory, 1 as special agreement, 2 as treaty.
        'juris_type': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                       0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                       1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                       0, 0, 1, 1, 2, 2, 2, 2, 1, 1,
                       2, 2, 2, 2, 2, 2, 0, 2, 1, 1,
                       2, 2, 2, 0, 2, 2, 2, 2, 2, 2,
                       0, 0, 0, 2],
        #this is with 0 compulsory, 1 as treaty, and 2 as special agreement
        'juris_type': [2, 0, 2, 2, 0, 0, 2, 2, 0, 0,
                       0, 0, 2, 2, 2, 2, 2, 2, 2, 0,
                       2, 2, 0, 2, 2, 2, 0, 2, 0, 0,
                       0, 0, 2, 2, 1, 1, 1, 1, 2, 2,
                       1, 1, 1, 1, 1, 1, 0, 1, 2, 2,
                       1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                       0, 0, 0, 1],
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
