import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ==========================================
# LOAD DATA
# ==========================================

reviews = pd.read_csv(DATA_DIR / "reviews.csv")
products = pd.read_csv(DATA_DIR / "products.csv")


# ==========================================
# DATA PREPARATION
# ==========================================

reviews["user_id"] = reviews["user_id"].astype(str)
reviews["product_id"] = reviews["product_id"].astype(str)

products["product_id"] = products["product_id"].astype(str)


# ==========================================
# CREATE USER-PRODUCT MATRIX
# ==========================================

user_product_matrix = reviews.pivot_table(
    index="user_id",
    columns="product_id",
    values="star_rating",
    aggfunc="mean",
    fill_value=0
)


# ==========================================
# ITEM SIMILARITY
# ==========================================

item_similarity = cosine_similarity(
    user_product_matrix.T
)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_product_matrix.columns,
    columns=user_product_matrix.columns
)


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend_products(user_id, top_n=5):

    user_id = str(user_id)

    if user_id not in user_product_matrix.index:
        print("Customer not found!")
        return pd.DataFrame()

    user_ratings = user_product_matrix.loc[user_id]

    # Products rated by customer
    rated_products = user_ratings[
        user_ratings > 0
    ].sort_values(ascending=False)

    recommendation_scores = {}

    # Find similar products
    for product_id, rating in rated_products.items():

        similar_products = item_similarity_df[
            product_id
        ].sort_values(
            ascending=False
        )

        for similar_product, similarity in similar_products.items():

            # Don't recommend products already rated
            if user_ratings[similar_product] > 0:
                continue

            score = rating * similarity

            if similar_product not in recommendation_scores:
                recommendation_scores[similar_product] = 0

            recommendation_scores[similar_product] += score

    # Sort recommendations
    recommendations = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    if not recommendations:
        return pd.DataFrame()

    result = pd.DataFrame(
        recommendations,
        columns=[
            "product_id",
            "recommendation_score"
        ]
    )

    # Add product information
    result = result.merge(
        products,
        on="product_id",
        how="left"
    )

    return result


# ==========================================
# TEST RECOMMENDATION
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 60)

    print("\nTotal Customers:",
          user_product_matrix.shape[0])

    print("Total Products:",
          user_product_matrix.shape[1])

    # Select first customer for testing
    test_user = user_product_matrix.index[0]

    print("\nSelected Customer:", test_user)

    recommendations = recommend_products(
        test_user,
        top_n=5
    )

    print("\nRecommended Products:")

    if recommendations.empty:

        print("No recommendations available.")

    else:

        display_columns = [
            "product_id",
            "recommendation_score"
        ]

        # Add available product columns
        for column in [
            "category",
            "price_usd",
            "avg_rating",
            "brand_tier"
        ]:

            if column in recommendations.columns:
                display_columns.append(column)

        print(
            recommendations[
                display_columns
            ].to_string(index=False)
        )