import pandas as pd
import numpy as np

# ==============================
# LOAD DATASETS
# ==============================

reviews = pd.read_csv("data/reviews.csv")
products = pd.read_csv("data/products.csv")
sellers = pd.read_csv("data/sellers.csv")
sentiment = pd.read_csv("data/sentiment_labels.csv")
monthly = pd.read_csv("data/monthly_trends.csv")

print("=" * 60)
print("SMART E-COMMERCE BIG DATA ANALYSIS")
print("=" * 60)

# ==============================
# DATASET SIZE
# ==============================

print("\nDATASET SIZES")

print("Reviews:", reviews.shape)
print("Products:", products.shape)
print("Sellers:", sellers.shape)
print("Sentiment:", sentiment.shape)
print("Monthly Trends:", monthly.shape)

# ==============================
# COLUMN INFORMATION
# ==============================

print("\nREVIEWS COLUMNS")
print(reviews.columns.tolist())

print("\nPRODUCTS COLUMNS")
print(products.columns.tolist())

print("\nSELLERS COLUMNS")
print(sellers.columns.tolist())

print("\nSENTIMENT COLUMNS")
print(sentiment.columns.tolist())

print("\nMONTHLY TRENDS COLUMNS")
print(monthly.columns.tolist())

# ==============================
# FIRST 5 RECORDS
# ==============================

print("\nFIRST 5 REVIEWS")
print(reviews.head())

print("\nFIRST 5 PRODUCTS")
print(products.head())

# ==============================
# BASIC INFORMATION
# ==============================

print("\nREVIEWS INFORMATION")
print(reviews.info())

print("\nPRODUCTS INFORMATION")
print(products.info())

# ==============================
# BASIC STATISTICS
# ==============================

print("\nREVIEW STATISTICS")
print(reviews.describe())

print("\nPRODUCT STATISTICS")
print(products.describe(include="all"))

# ==============================
# MISSING VALUES
# ==============================

print("\nMISSING VALUES - REVIEWS")
print(reviews.isnull().sum())

print("\nMISSING VALUES - PRODUCTS")
print(products.isnull().sum())

# ==============================
# DUPLICATE VALUES
# ==============================

print("\nDUPLICATE RECORDS")

print("Reviews duplicates:", reviews.duplicated().sum())
print("Products duplicates:", products.duplicated().sum())
print("Sellers duplicates:", sellers.duplicated().sum())
print("Sentiment duplicates:", sentiment.duplicated().sum())

print("\nDATASET ANALYSIS COMPLETED!")