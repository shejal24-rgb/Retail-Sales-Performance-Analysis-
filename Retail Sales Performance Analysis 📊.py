import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("retail_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.drop_duplicates()

df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Unit_Price"] = df["Unit_Price"].fillna(df["Unit_Price"].median())
df["Discount"] = df["Discount"].fillna(0)
df["Revenue"] = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount"])
df["Month"] = df["Date"].dt.strftime("%b")

print("Dataset Overview")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nTotal Revenue:", round(df["Revenue"].sum(), 2))
print("Average Revenue:", round(df["Revenue"].mean(), 2))
print("Total Orders:", df["Order_ID"].nunique())

category_sales = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
print("\nCategory-wise Sales")
print(category_sales)

product_sales = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
print("\nProduct-wise Sales")
print(product_sales)

region_sales = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
print("\nRegion-wise Sales")
print(region_sales)

monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
print("\nMonthly Sales")
print(monthly_sales)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Category-wise Revenue")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_sales.index.astype(str), y=monthly_sales.values, marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Product-wise Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title("Region-wise Revenue")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Revenue"], bins=10, kde=True)
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.tight_layout()
plt.show()

print("\nBusiness Insights")
print("Top Category:", category_sales.idxmax())
print("Top Product:", product_sales.idxmax())
print("Top Region:", region_sales.idxmax())
print("Best Sales Month:", monthly_sales.idxmax())
