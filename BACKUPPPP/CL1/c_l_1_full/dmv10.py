# ----------------------------------------------------------
# Data Wrangling on Real Estate Market
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Load and Clean Data
df = pd.read_csv("RealEstate_Prices.csv")
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
)

print("✅ Data Loaded Successfully\n")
print(df.info(), "\n")
print("Missing Values:\n", df.isnull().sum(), "\n")

# 2. Handle Missing Values
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# 3. Parse and Convert Total Sqft
def to_sqft(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if "-" in s:
        parts = s.split("-")
        return np.mean([float(p) for p in parts]) if all(p.replace('.', '', 1).isdigit() for p in parts) else np.nan
    if "acre" in s.lower():
        try: return float(s.split()[0]) * 43560
        except: return np.nan
    try: return float(s.replace(",", ""))
    except: return np.nan

if "total_sqft" in df.columns:
    df["total_sqft_num"] = df["total_sqft"].apply(to_sqft)

# 4. Filter Example (only “Super built-up Area”)
if "area_type" in df.columns:
    df = df[df["area_type"].str.lower().str.contains("super built-up")]

# 5. Encode Categorical Variables
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# 6. Aggregate and Summarize
if "price" in df.columns and "location" in df.columns:
    avg_price = df.groupby("location")["price"].mean().reset_index().sort_values(by="price", ascending=False)
    print("\n🏙️ Average Price by Location:\n", avg_price.head())

# 7. Handle Outliers
for col in num_cols:
    lower, upper = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lower, upper)

# 8. Visualization Dashboard
plt.figure(figsize=(18, 10))
plt.suptitle("Real Estate Market Insights", fontsize=16, fontweight='bold')

plt.subplot(2, 2, 1)
sns.histplot(df["price"], kde=True, color="skyblue")
plt.title("Distribution of Price")

if "total_sqft_num" in df.columns:
    plt.subplot(2, 2, 2)
    sns.scatterplot(x="total_sqft_num", y="price", data=df, alpha=0.7)
    plt.title("Price vs Total Sqft")

if "location" in df.columns:
    top_locs = df["location"].value_counts().index[:5]
    plt.subplot(2, 2, 3)
    sns.boxplot(x="location", y="price", data=df[df["location"].isin(top_locs)])
    plt.title("Price by Top 5 Locations")
    plt.xticks(rotation=30)

plt.subplot(2, 2, 4)
corr = df[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# 9. Export Cleaned Dataset
df.to_csv("RealEstate_Prices_Cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as 'RealEstate_Prices_Cleaned.csv'")
