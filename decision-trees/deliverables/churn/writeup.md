# Telco Churn: Short Writeup

## Exploratory analysis (two features)
- **Contract:** Month-to-month customers show a much higher churn fraction than one-year and two-year contracts.
- **SeniorCitizen:** Senior-citizen customers generally show higher churn fraction than non-senior customers.

## Model
- Pipeline used: label encoding, selected-feature reduction, MinMax scaling for tenure/MonthlyCharges/TotalCharges, SMOTE balancing on training data.
- Classifiers compared: XGBClassifier, LGBMClassifier, RandomForestClassifier, DecisionTreeClassifier, and a StackingClassifier.
- `customerID` was excluded from modeling to avoid leakage.
- Data split: 80% train / 20% test.

## Evaluation
- Best model: **LGBMClassifier**
- Precision (churn class): **0.546**
- Recall (churn class): **0.826**
- F1 (churn class): **0.657**
- ROC-AUC (probability-based): **0.861**
- Confusion matrix saved as `plots/03_confusion_matrix.png`.
- ROC curve saved as `plots/04_roc_curve_best_model.png`.

## Validation set?
- A separate validation split is optional here because repeated stratified cross-validation was used for model comparison on training data.
- For heavy hyperparameter tuning, use train/validation/test or nested CV.

## Recommended action
- Prioritize retention offers for month-to-month customers and improve support/onboarding in the first months.
- Target high-risk segments (e.g., senior-citizen month-to-month customers) with proactive outreach and incentive plans.
