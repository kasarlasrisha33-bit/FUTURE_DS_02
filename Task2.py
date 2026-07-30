import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.describe(include="object"))
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].isnull().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print(df["Churn"].value_counts())
df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn")

plt.xlabel("Churn")

plt.ylabel("Customers")

plt.show()
df["MonthlyCharges"].plot(kind="hist", bins=20)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.show()
# Tenure Histogram
df["tenure"].plot(kind="hist", bins=20)
plt.title("Tenure Distribution")
plt.xlabel("Tenure")
plt.show()
# Churn Count
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Customers")
plt.show()
df.groupby("gender")["Churn"].value_counts().unstack().plot(kind="bar")
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Customers")
plt.show()
df.groupby("SeniorCitizen")["Churn"].value_counts().unstack().plot(kind="bar")
plt.title("Churn by Senior Citizen")
plt.xlabel("Senior Citizen")
plt.ylabel("Customers")
plt.show()
import matplotlib.pyplot as plt

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(6,4))
plt.imshow(numeric_df.corr(), cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=90)
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
plt.title("Correlation Heatmap")
plt.show()
