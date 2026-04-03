We can use Chi-squared for caregorical and ANOVA for numerical.
Based on this statistical test, tenure, monthly charges, totalCharges, senior citizens,contract, payment methods and online security are the important features. The rest of the features aren't statistically significant for the output hence we dropped them.


The dataset has far more churn customers than the non-churn customers; we use SMOTE to artificially generate more churn sampeles; we use treebased classifiers like XGBoost, Random Foresct Classifier, Decision Tree Classifier etc. I stacked the classifiers and achieved an accuracy of 0.8 with reasonable precision, recall and f1 score





Combined all four of the models above into one "super model," using LightGBM as the final decision-maker. This stacked model achieved a Cross-Validation score of ~90.8% and an F1 Score of 83%.













Entropy INformation gain
Huffman codes, string of bits, using a particular set of codes; decode
Small dataset, split, calculate the entropy and infrmatoon gain

Probabiluty distribution huffman codes out of that distribution