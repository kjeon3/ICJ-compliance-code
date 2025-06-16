import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

def compute_model_fit_stats(X_train, X_test, y_train, y_test):
    model = LogisticAT()
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)

    # Fit null model (intercept only)
    X_null = np.ones((len(y_test), 1))
    null_model = LogisticAT()
    null_model.fit(np.ones((len(y_train), 1)), y_train)
    y_null_proba = null_model.predict_proba(X_null)

    # Log likelihoods
    LL_full = -log_loss(y_test, y_pred_proba, labels=np.unique(y_train), normalize=False)
    LL_null = -log_loss(y_test, y_null_proba, labels=np.unique(y_train), normalize=False)

    neg2LL = -2 * LL_full
    N = len(y_test)
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
        'compliance_level': [1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                             1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                             0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                             0, 1,],
        'judge_from_country': [0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1],
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

    y = data['compliance_level']
    X = data.drop(columns=['compliance_level'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔀 Random split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    stats, model = compute_model_fit_stats(X_train, X_test, y_train, y_test)

    print("\n📊 Test Set Model Fit Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Evaluate prediction accuracy
    y_pred = model.predict(X_test)
    y_true = y_test.to_numpy()

    compliance_mask = (y_true == 0)
    noncompliance_mask = (y_true == 1)

    compliance_correct = np.sum((y_true == 0) & (y_pred == 0))
    noncompliance_correct = np.sum((y_true == 1) & (y_pred == 1))

    compliance_accuracy = compliance_correct / np.sum(compliance_mask) * 100
    noncompliance_accuracy = noncompliance_correct / np.sum(noncompliance_mask) * 100
    overall_accuracy = np.mean(y_true == y_pred) * 100

    print("\n✅ Classification Accuracy on Test Set:")
    print(f"% Compliance (0) correctly predicted: {compliance_accuracy:.1f}%")
    print(f"% Noncompliance (1) correctly predicted: {noncompliance_accuracy:.1f}%")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
