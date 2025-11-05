# Ensemble Boosting Algorithms on Iris Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define models
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Step 3: Train models
models = {'AdaBoost': ada, 'GBM': gbm, 'XGBoost': xgb}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 3),
        'Precision': round(precision_score(y_test, y_pred, average='macro'), 3),
        'Recall': round(recall_score(y_test, y_pred, average='macro'), 3),
        'F1-Score': round(f1_score(y_test, y_pred, average='macro'), 3)
    })

# Step 4: Voting Classifier (Soft Voting)
voting = VotingClassifier(
    estimators=[('AdaBoost', ada), ('GBM', gbm), ('XGBoost', xgb)],
    voting='soft'
)
voting.fit(X_train, y_train)
y_vote = voting.predict(X_test)
results.append({
    'Model': 'Voting Ensemble',
    'Accuracy': round(accuracy_score(y_test, y_vote), 3),
    'Precision': round(precision_score(y_test, y_vote, average='macro'), 3),
    'Recall': round(recall_score(y_test, y_vote, average='macro'), 3),
    'F1-Score': round(f1_score(y_test, y_vote, average='macro'), 3)
})

# Step 5: Results Table
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n", results_df)

# Step 6: Visualization on one page
plt.figure(figsize=(10, 4))

# Bar chart of accuracies
plt.subplot(1, 2, 1)
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis')
plt.title('Accuracy Comparison')
plt.ylim(0.9, 1.0)

# Confusion matrix for best model
best_model = xgb
y_pred_best = best_model.predict(X_test)
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='coolwarm')
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.suptitle('Boosting Algorithms & Voting Mechanism on Iris Dataset', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
