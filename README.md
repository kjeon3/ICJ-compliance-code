My pvalues.py file contains the code to obtain the p-values for each of my independent variables. 


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



    results = find_best_combinations_statsmodels(data, target="compliance_level")
    if results:
        results_df = pd.DataFrame(results).sort_values(by="P-value", na_position="last")
        print("\nOrdinal Regression Summary (statsmodels):")
        print(results_df.to_string(index=False))
    else:
        print("No valid models returned.")

    
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

