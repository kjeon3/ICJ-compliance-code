import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

def compute_model_fit_stats(X, y):
    model = LogisticAT()
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)

    # Fit null model
    X_null = np.ones((len(y), 1))
    null_model = LogisticAT()
    null_model.fit(X_null, y)
    y_null_proba = null_model.predict_proba(X_null)

    # Log likelihoods
    LL_full = -log_loss(y, y_pred_proba, labels=np.unique(y), normalize=False)
    LL_null = -log_loss(y, y_null_proba, labels=np.unique(y), normalize=False)

    neg2LL = -2 * LL_full
    N = len(y)
    R2_cox_snell = 1 - np.exp((LL_null - LL_full) * 2 / N)

    exponent = 2 * LL_null / N
    max_exponent = 700
    safe_exponent = min(exponent, max_exponent)
    denom = 1 - np.exp(safe_exponent)
    epsilon = 1e-10
    denom = max(denom, epsilon)
    R2_nagelkerke = R2_cox_snell / denom

    pseudo_r2 = 1 - (LL_full / LL_null)

    return {
        "-2 Log Likelihood": round(neg2LL, 3),
        "Cox & Snell R²": round(R2_cox_snell, 3),
        "Nagelkerke R²": round(R2_nagelkerke, 3),
        "McFadden’s Pseudo R²": round(pseudo_r2, 3)
    }, model


if __name__ == "__main__":
    data = pd.DataFrame({
        'compliance_level': [2, 0, 0, 0, 1, 1, 0, 0, 0, 2,
                             2, 1, 0, 0, 0, 0, 0, 0, 0, 2,
                             0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                             0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 2, 0, 1, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0, 0, 2,
                             0, 1, 1, 2],
        #from here these are my independent variables.

        'Log_GDP_billions': [8, 10.6259, 10.642, 10.1644, 8.3802, 9.5211, 11.3979, 11.3979, 10.854, 9.1818,
                             9.1818, 11.0022, 10.5284, 12.6064, 11.5523, 10.4987, 9.0453, 9.3096, 9.2672, 12.7896,
                             9.6937, 9.7642, 11.0813, 10.4565, 9.5877, 10.244, 9.9533, 13.0253, 11.2915, 10.98,
                             11.4123, 13.0864, 9.8176, 9.6415, 10.0924, 9.8704, 10.6521, 9.5658, 12.4669, 11.3633,
                             11.287, 9.9191, 10.6227, 11.4521, 10.247, 11.569, 10.467, 12.3222, 10.1284, 10.0095,
                             11.6236, 11.3027, 11.4142, 12.6902, 10.7516, 10.1149, 10.7954, 10.1149, 10.1149, 11.5064,
                             9.9768, 11.0402, 10.6587, 11.5382],
        'judge_from_country': [0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1],

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

    y = data['compliance_level']
    X = data.drop(columns=['compliance_level'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stats, model = compute_model_fit_stats(X_scaled, y)

    print("\n📊 Full Model Fit Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nCompliance accurately predicted percentages below:")

    y_pred = model.predict(X_scaled)
    y_true = y.to_numpy()

    # Accuracy breakdown
    compliance_mask = (y_true == 0)
    noncompliance_mask = (y_true == 1)
    #grouped_mask = (y_true >= 1)

    compliance_correct = np.sum((y_true == 0) & (y_pred == 0))
    noncompliance_correct = np.sum((y_true == 1) & (y_pred == 1))
    #grouped_correct = np.sum((y_true >= 1) & (y_pred >= 1))

    compliance_accuracy = compliance_correct / np.sum(compliance_mask) * 100
    noncompliance_accuracy = noncompliance_correct / np.sum(noncompliance_mask) * 100
    #grouped_accuracy = grouped_correct / np.sum(grouped_mask) * 100
    overall_accuracy = np.mean(y_true == y_pred) * 100

    print(f"% Compliance (0) correctly predicted: {compliance_accuracy:.1f}%")
    print(f"% Noncompliance (2) correctly predicted: {noncompliance_accuracy:.1f}%")
    #print(f"% Partial/Noncompliance (1 or 2) correctly grouped: {grouped_accuracy:.1f}%")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
