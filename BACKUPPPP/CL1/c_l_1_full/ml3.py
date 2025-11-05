# SVM for Handwritten Digit Classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
digits = datasets.load_digits()

# Step 2: Split into features and labels
X, y = digits.data, digits.target

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5: Train SVM model
svm_model = SVC(kernel='rbf', gamma=0.001, C=10)
svm_model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = svm_model.predict(X_test)

# Step 7: Evaluation
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8 & 9 Combined — Confusion Matrix + Sample Predictions on one page
plt.figure(figsize=(10, 6))

# Confusion Matrix (Left)
plt.subplot(2, 3, (1, 3))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Sample Predictions (Right)
for i in range(3):
    plt.subplot(2, 3, 4 + i)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    plt.axis('off')

plt.suptitle("SVM for Handwritten Digit Classification", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
