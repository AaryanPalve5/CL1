# ----------------------------
# Telecom Customer Churn - Data Cleaning & Preparation
# ----------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv('Telecom_Customer_Churn.csv')

# Normalize column names to lowercase
df.columns = df.columns.str.lower()

# ----------------------------
# 2. Explore data
# ----------------------------
print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# ----------------------------
# 3. Data Cleaning
# ----------------------------
# Convert TotalCharges to numeric
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].median())

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Remove duplicates
df = df.drop_duplicates()

# Standardize categorical values
for col in cat_cols:
    df[col] = df[col].str.lower().str.strip()

# Map churn to numeric
df['churn'] = df['churn'].map({'yes':1, 'no':0})

# ----------------------------
# 4. Outlier Handling
# ----------------------------
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# ----------------------------
# 5. Feature Engineering
# ----------------------------
if 'totalcharges' in df.columns and 'tenure' in df.columns:
    df['avg_monthly_charge'] = df['totalcharges'] / (df['tenure']+1)

# ----------------------------
# 6. Normalize / Scale numeric data
# ----------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ----------------------------
# 7. Train-test split
# ----------------------------
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 8. Visualizations on same page
# ----------------------------
fig, axes = plt.subplots(2,3, figsize=(18,10))

# Numeric feature distributions
for i, col in enumerate(numeric_cols[:3]):
    sns.histplot(df[col], ax=axes[0,i], kde=True)
    axes[0,i].set_title(f'Distribution of {col}')

# Churn count
sns.countplot(x='churn', data=df, ax=axes[1,0])
axes[1,0].set_title('Churn Count')

# Correlation heatmap
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')

# Boxplot: tenure by churn
sns.boxplot(x='churn', y='tenure', data=df, ax=axes[1,2])
axes[1,2].set_title('Tenure by Churn')

plt.tight_layout()
plt.show()

# ----------------------------
# 9. Export cleaned dataset
# ----------------------------
df.to_csv('Telecom_Customer_Churn_Cleaned.csv', index=False)
print("Cleaned dataset saved as 'Telecom_Customer_Churn_Cleaned.csv'")
