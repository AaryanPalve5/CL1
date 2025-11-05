import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("Retail_Sales_Data.csv")

# Add a 'Region' column randomly for demonstration
regions = ["North", "South", "East", "West"]
df["Region"] = np.random.choice(regions, size=len(df))

# Rename column for clarity
df.rename(columns={"Total Amount": "Sales_Amount"}, inplace=True)

# Group by region
region_sales = df.groupby("Region")["Sales_Amount"].sum().sort_values(ascending=False)
print("\nTotal Sales by Region:\n", region_sales)

# Bar plot for region sales
plt.figure(figsize=(8,5))
region_sales.plot(kind="bar", color="teal")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart
plt.figure(figsize=(6,6))
region_sales.plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
plt.title("Sales Distribution by Region")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Top performing region
top_region = region_sales.idxmax()
print(f"\n🏆 Top-Performing Region: {top_region} with Sales = {region_sales.max()}")

# Group by region and product category
region_category_sales = df.groupby(["Region", "Product Category"])["Sales_Amount"].sum().unstack().fillna(0)
print("\nSales by Region and Product Category:\n", region_category_sales)

# Stacked bar chart
region_category_sales.plot(kind="bar", stacked=True, figsize=(10,6), colormap="tab10")
plt.title("Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.legend(title="Product Category")
plt.tight_layout()
plt.show()

print("\n✅ Data Aggregation & Visualization complete!")
