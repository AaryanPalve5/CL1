import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import seaborn as sns

# ----------------------------
# 1. Load Sample Dataset from Seaborn
# ----------------------------
tips = sns.load_dataset('tips')  # built-in dataset
print("Original Dataset Head:\n", tips.head())

# Simulate multiple file formats
tips.to_csv('sample_sales.csv', index=False)
tips.to_excel('sample_sales.xlsx', index=False)
tips.to_json('sample_sales.json', orient='records')

# Load data from the simulated files
csv_data = pd.read_csv('sample_sales.csv')
excel_data = pd.read_excel('sample_sales.xlsx')
json_data = pd.read_json('sample_sales.json')

# ----------------------------
# 2. Explore Data
# ----------------------------
print("CSV Head:\n", csv_data.head())
print("Excel Head:\n", excel_data.head())
print("JSON Head:\n", json_data.head())

print("Missing Values CSV:\n", csv_data.isnull().sum())

# ----------------------------
# 3. Data Cleaning
# ----------------------------
# No missing values in this dataset for demo, but example:
csv_data.fillna(0, inplace=True)
excel_data.fillna(0, inplace=True)
json_data.fillna(0, inplace=True)

# ----------------------------
# 4. Combine Data
# ----------------------------
csv_data.columns = [col.lower() for col in csv_data.columns]
excel_data.columns = [col.lower() for col in excel_data.columns]
json_data.columns = [col.lower() for col in json_data.columns]

combined_data = pd.concat([csv_data, excel_data, json_data], ignore_index=True)

# ----------------------------
# 5. Data Transformation
# ----------------------------
# Let's treat 'total_bill' as total sales and 'tip' as profit
combined_data['total_sales'] = combined_data['total_bill']
combined_data['profit'] = combined_data['tip']

# Derive new metric: average order value (here, total_bill per customer)
combined_data['avg_order_value'] = combined_data['total_sales']

# ----------------------------
# 6. Data Analysis
# ----------------------------
print("Descriptive Statistics:\n", combined_data.describe())

# Total sales by day
total_sales_day = combined_data.groupby('day')['total_sales'].sum()
print("Total Sales by Day:\n", total_sales_day)

# Average order value
print("Average Order Value:", combined_data['avg_order_value'].mean())

# ----------------------------
# 7. Visualizations (All on one page)
# ----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns

# Bar plot: Total sales by day
total_sales_day.plot(kind='bar', ax=axes[0], title='Total Sales by Day')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Total Sales')

# Pie chart: distribution of payment by sex
combined_data['sex'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', ax=axes[1], title='Customer Gender Distribution'
)
axes[1].set_ylabel('')

# Box plot: Average order value by day
sns.boxplot(x='day', y='avg_order_value', data=combined_data, ax=axes[2])
axes[2].set_title('Average Order Value by Day')

plt.tight_layout()
plt.show()

