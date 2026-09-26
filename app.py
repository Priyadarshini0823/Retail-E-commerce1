import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Import existing recommendation model
from recommendation import recommend_products


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS - CRIMSON & OBSIDIAN DARK THEME
# =========================================================

st.markdown(
    """
    <style>

    /* Main App & Obsidian Background */
    .stApp {
        background-color: #0b0b10;
        color: #e2e8f0;
    }

    .main {
        background-color: #0b0b10;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #11070e 0%,
            #1c0915 50%,
            #11070e 100%
        );
        border-right: 1px solid #7f1d1d;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 20px;
    }

    .sidebar-logo {
        text-align: center;
        padding: 10px 5px 20px 5px;
    }

    .sidebar-logo-icon {
        font-size: 38px;
        margin-bottom: 5px;
        filter: drop-shadow(0 0 10px rgba(220, 38, 38, 0.6));
    }

    .sidebar-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }

    .sidebar-subtitle {
        color: #f87171;
        font-size: 11px;
        margin-top: 4px;
        font-weight: 600;
    }

    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #991b1b, transparent);
        margin: 10px 0 20px 0;
    }

    .nav-title {
        color: #cbd5e1;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Sidebar Radio Buttons */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }

    [data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
        font-size: 14px !important;
        padding: 9px 12px !important;
        border-radius: 9px !important;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }

    [data-testid="stSidebar"] label:hover {
        background-color: #270e1a !important;
        color: #ffffff !important;
        border-color: #991b1b !important;
    }

    /* Headers */
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 3px;
        letter-spacing: -0.5px;
    }

    .main-title span {
        color: #ef4444;
    }

    .main-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 20px;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #14141f,
            #1a1424
        );
        border: 1px solid #7f1d1d;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(153, 27, 27, 0.15);
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 13px !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            90deg,
            #991b1b,
            #dc2626
        );
        color: white;
        border: 1px solid #ef4444;
        border-radius: 9px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #b91c1c,
            #ef4444
        );
        border-color: #f87171;
        color: white;
        box-shadow: 0 4px 18px rgba(239, 68, 68, 0.4);
    }

    /* Tables & Inputs */
    [data-testid="stDataFrame"] {
        border: 1px solid #371726;
        border-radius: 10px;
    }

    div[data-baseweb="select"] > div {
        background-color: #14141f !important;
        border-color: #7f1d1d !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #120b14;
        padding: 7px;
        border-radius: 10px;
        border: 1px solid #450a0a;
    }

    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
        border-radius: 7px;
        padding: 8px 15px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #801328;
        color: #ffffff !important;
    }

    /* Custom Card Containers */
    .custom-card {
        background-color: #14141f;
        border: 1px solid #7f1d1d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA LOADING WITH RELATIVE PATHS
# =========================================================

@st.cache_data
def load_data():
    try:
        reviews = pd.read_csv(DATA_DIR / "reviews.csv")
        products = pd.read_csv(DATA_DIR / "products.csv")
        sellers = pd.read_csv(DATA_DIR / "sellers.csv")
        sentiment = pd.read_csv(DATA_DIR / "sentiment_labels.csv")
        monthly = pd.read_csv(DATA_DIR / "monthly_trends.csv")

        # Standardize ID column types for clean joins and comparisons
        reviews["user_id"] = reviews["user_id"].astype(str)
        reviews["product_id"] = reviews["product_id"].astype(str)
        products["product_id"] = products["product_id"].astype(str)
        sellers["seller_id"] = sellers["seller_id"].astype(str)
        if "seller_id" in products.columns:
            products["seller_id"] = products["seller_id"].astype(str)

        return reviews, products, sellers, sentiment, monthly
    except Exception as e:
        st.error(f"Error loading CSV datasets from relative path data/: {e}")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


reviews, products, sellers, sentiment, monthly = load_data()

if reviews.empty or products.empty:
    st.error("Primary datasets (reviews.csv, products.csv) could not be loaded. Please ensure data/ folder exists.")
    st.stop()


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

if "cart" not in st.session_state:
    st.session_state.cart = []

if "wishlist" not in st.session_state:
    st.session_state.wishlist = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None

if "applied_promo" not in st.session_state:
    st.session_state.applied_promo = None


# =========================================================
# CALCULATE GLOBAL AGGREGATIONS
# =========================================================

rating_counts = (
    reviews["star_rating"]
    .value_counts()
    .sort_index()
    .reset_index()
)
rating_counts.columns = ["Rating", "Count"]


# =========================================================
# SIDEBAR NAVIGATION & SESSION WIDGETS
# =========================================================

st.sidebar.markdown(
    """<div class="sidebar-logo">
    <div class="sidebar-logo-icon">🛒</div>
    <div class="sidebar-title">Smart E-Commerce</div>
    <div class="sidebar-subtitle">Analytics & Recommendation</div>
</div>
<div class="sidebar-divider"></div>
<div class="nav-title">Navigation</div>""",
    unsafe_allow_html=True
)

cart_count = sum(int(item.get("quantity", 1)) for item in st.session_state.cart)
wish_count = len(st.session_state.wishlist)
feed_count = len(st.session_state.feedback)

cart_badge = f" ({cart_count})" if cart_count > 0 else ""
wish_badge = f" ({wish_count})" if wish_count > 0 else ""
feed_badge = f" ({feed_count})" if feed_count > 0 else ""

page = st.sidebar.radio(
    "",
    [
        "🏠  Dashboard",
        "📊  Big Data Analytics",
        "👤  Customer Analysis",
        "🤖  Recommendations",
        f"🛒  Cart{cart_badge}",
        f"❤️  Wishlist{wish_badge}",
        f"💬  Feedback{feed_badge}"
    ],
    label_visibility="collapsed"
)

if "Dashboard" in page:
    page_key = "Dashboard"
elif "Big Data Analytics" in page:
    page_key = "Big Data Analytics"
elif "Customer Analysis" in page:
    page_key = "Customer Analysis"
elif "Recommendations" in page:
    page_key = "Recommendations"
elif "Cart" in page:
    page_key = "Cart"
elif "Wishlist" in page:
    page_key = "Wishlist"
elif "Feedback" in page:
    page_key = "Feedback"
else:
    page_key = "Dashboard"

# Session Summary Widget
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='nav-title'>Session Summary</div>", unsafe_allow_html=True)
st.sidebar.caption(f"🛒 Cart Items: {cart_count}")
st.sidebar.caption(f"❤️ Wishlist Items: {wish_count}")
st.sidebar.caption(f"💬 Feedback Logged: {feed_count}")

if st.session_state.applied_promo:
    st.sidebar.caption(f"🏷️ Promo Code: `{st.session_state.applied_promo}`")

if st.session_state.feedback:
    feedback_df = pd.DataFrame(st.session_state.feedback)
    feedback_csv = feedback_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "📥 Download Feedback CSV",
        data=feedback_csv,
        file_name="feedback_summary.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🛒 Smart <span>E-Commerce</span> Analytics
    </div>
    <div class="main-subtitle">
        Big Data Analytics & Intelligent Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# 1. DASHBOARD
# =========================================================

if page_key == "Dashboard":

    st.header("📊 E-Commerce Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📝 Total Reviews",
            f"{len(reviews):,}"
        )

    with col2:
        st.metric(
            "👥 Total Customers",
            f"{reviews['user_id'].nunique():,}"
        )

    with col3:
        st.metric(
            "📦 Total Products",
            f"{products['product_id'].nunique():,}"
        )

    with col4:
        avg_rating_val = reviews["star_rating"].mean()
        st.metric(
            "⭐ Average Rating",
            f"{avg_rating_val:.2f} / 5.0"
        )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⭐ Customer Rating Distribution")
        fig_ratings = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            title="Rating Count Distribution",
            template="plotly_dark",
            color="Count",
            color_continuous_scale="Reds"
        )
        fig_ratings.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    with col_right:
        if "category" in products.columns:
            st.subheader("🛍️ Top Product Categories")
            cat_counts = (
                products["category"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            cat_counts.columns = ["Category", "Products"]

            fig_cats = px.bar(
                cat_counts,
                x="Category",
                y="Products",
                title="Products per Category",
                template="plotly_dark",
                color="Products",
                color_continuous_scale="Purples"
            )
            fig_cats.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cats, use_container_width=True)

    # Monthly Trend Analysis if monthly dataset is present
    if not monthly.empty and "period" in monthly.columns and "review_count" in monthly.columns:
        st.markdown("---")
        st.subheader("📈 Monthly Review Volume & Trends")
        fig_trend = px.line(
            monthly,
            x="period",
            y="review_count",
            title="Monthly Review Volume",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=["#ef4444"]
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# =========================================================
# 2. BIG DATA ANALYTICS
# =========================================================

elif page_key == "Big Data Analytics":

    st.header("📊 Big Data Analytics")

    tab1, tab2, tab3 = st.tabs(
        [
            "⭐ Rating Analysis",
            "📦 Product Analytics",
            "🏪 Seller Analytics"
        ]
    )

    # ---------------- TAB 1: RATINGS ----------------
    with tab1:
        st.subheader("⭐ Comprehensive Rating Statistics")

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.write("##### Summary Statistics")
            r_stats = reviews["star_rating"].describe().to_frame("Value")
            st.dataframe(r_stats, use_container_width=True)

        with col_b:
            fig_pie = px.pie(
                rating_counts,
                names="Rating",
                values="Count",
                title="⭐ Rating Percentage Share",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Rating Frequency Breakdown")
        fig_freq = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            text="Count",
            title="Total Reviews by Rating Star",
            template="plotly_dark",
            color="Count",
            color_continuous_scale="Reds"
        )
        fig_freq.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # ---------------- TAB 2: PRODUCTS ----------------
    with tab2:
        st.subheader("📦 Top Product Performance")

        product_columns = [
            "product_id",
            "category",
            "price_usd",
            "total_reviews",
            "avg_rating",
            "brand_tier"
        ]
        avail_p_cols = [c for c in product_columns if c in products.columns]

        prod_sorted = (
            products[avail_p_cols]
            .sort_values(by="total_reviews", ascending=False)
            .head(20)
        )
        st.dataframe(prod_sorted, use_container_width=True)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            if "avg_rating" in products.columns and "total_reviews" in products.columns:
                top_rated_prods = products.sort_values(by="total_reviews", ascending=False).head(10)
                fig_prod_rating = px.bar(
                    top_rated_prods,
                    x="product_id",
                    y="avg_rating",
                    title="⭐ Average Rating of Top 10 Reviewed Products",
                    template="plotly_dark",
                    color="avg_rating",
                    color_continuous_scale="Reds"
                )
                fig_prod_rating.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_prod_rating, use_container_width=True)

        with col_p2:
            if "price_usd" in products.columns and "avg_rating" in products.columns:
                fig_scatter = px.scatter(
                    products.head(300),
                    x="price_usd",
                    y="avg_rating",
                    color="category" if "category" in products.columns else None,
                    title="💰 Price vs Average Rating Sample",
                    template="plotly_dark"
                )
                fig_scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------------- TAB 3: SELLERS ----------------
    with tab3:
        st.subheader("🏪 Seller Performance Overview")

        if not sellers.empty:
            seller_cols = [
                "seller_id",
                "n_products",
                "avg_product_rating",
                "total_reviews",
                "avg_shipping_days",
                "return_rate_pct",
                "ships_internationally"
            ]
            avail_s_cols = [c for c in seller_cols if c in sellers.columns]

            top_sellers = (
                sellers[avail_s_cols]
                .sort_values(by="total_reviews" if "total_reviews" in sellers.columns else sellers.columns[0], ascending=False)
                .head(20)
            )
            st.dataframe(top_sellers, use_container_width=True)

            if "total_reviews" in sellers.columns and "seller_id" in sellers.columns:
                fig_sellers = px.bar(
                    top_sellers.head(10),
                    x="seller_id",
                    y="total_reviews",
                    title="🏪 Top Sellers by Total Reviews",
                    template="plotly_dark",
                    color="total_reviews",
                    color_continuous_scale="Purples"
                )
                fig_sellers.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_sellers, use_container_width=True)
        else:
            st.info("No seller data available.")


# =========================================================
# 3. CUSTOMER ANALYSIS
# =========================================================

elif page_key == "Customer Analysis":

    st.header("👤 Customer Behavior & History Analysis")

    customer_ids = reviews["user_id"].astype(str).unique().tolist()
    customer_ids.sort()

    default_idx = 0
    if st.session_state.selected_customer in customer_ids:
        default_idx = customer_ids.index(st.session_state.selected_customer)

    selected_cust = st.selectbox(
        "👤 Select Customer ID to Analyze",
        customer_ids,
        index=default_idx
    )
    st.session_state.selected_customer = selected_cust

    cust_reviews = reviews[reviews["user_id"].astype(str) == selected_cust]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("📝 Total Reviews Written", len(cust_reviews))

    with c2:
        cust_avg_r = cust_reviews["star_rating"].mean() if not cust_reviews.empty else 0.0
        st.metric("⭐ Customer Average Rating", f"{cust_avg_r:.2f}")

    with c3:
        unique_prods = cust_reviews["product_id"].nunique() if not cust_reviews.empty else 0
        st.metric("📦 Unique Products Reviewed", unique_prods)

    st.markdown("---")
    st.subheader(f"📜 Review History for Customer `{selected_cust}`")

    hist_cols = [
        c for c in [
            "review_id",
            "product_id",
            "star_rating",
            "review_date",
            "sentiment",
            "price_usd",
            "helpful_votes"
        ] if c in cust_reviews.columns
    ]

    st.dataframe(cust_reviews[hist_cols], use_container_width=True)

    if not cust_reviews.empty:
        st.markdown("---")
        st.subheader("⭐ Customer Rating Pattern")

        cust_rating_dist = (
            cust_reviews["star_rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        cust_rating_dist.columns = ["Rating", "Count"]

        fig_cust_p = px.bar(
            cust_rating_dist,
            x="Rating",
            y="Count",
            title=f"Rating Pattern for {selected_cust}",
            template="plotly_dark",
            color="Count",
            color_continuous_scale="Reds"
        )
        fig_cust_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cust_p, use_container_width=True)


# =========================================================
# 4. PRODUCT RECOMMENDATIONS
# =========================================================

elif page_key == "Recommendations":

    st.header("🤖 Personalized Product Recommendations")
    st.write("Select a customer, adjust recommendation filters, and generate AI-driven personalized product recommendations.")

    customer_ids = reviews["user_id"].astype(str).unique().tolist()
    customer_ids.sort()

    default_idx = 0
    if st.session_state.selected_customer in customer_ids:
        default_idx = customer_ids.index(st.session_state.selected_customer)

    selected_cust = st.selectbox(
        "👤 Select Target Customer",
        customer_ids,
        index=default_idx
    )
    st.session_state.selected_customer = selected_cust

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        cat_list = ["All"]
        if "category" in products.columns:
            cat_list += sorted(products["category"].dropna().astype(str).unique().tolist())
        sel_cat = st.selectbox("🛍️ Category Filter", cat_list)

    with col_f2:
        if "price_usd" in products.columns and not products["price_usd"].dropna().empty:
            max_p_val = float(products["price_usd"].max())
            price_limit = st.number_input(
                "💰 Maximum Price ($)",
                min_value=0.0,
                max_value=10000.0,
                value=max_p_val,
                step=10.0
            )
        else:
            price_limit = None

    with col_f3:
        min_rating_val = st.slider(
            "⭐ Minimum Rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5
        )

    top_n_count = st.slider("🔢 Number of Recommendations", 1, 20, 5)

    if st.button("🚀 Generate Recommendations", use_container_width=True):
        if not selected_cust:
            st.error("Please select a valid customer.")
        else:
            with st.spinner("Executing recommendation algorithm..."):
                try:
                    # Fetch broader set to allow client-side filtering
                    raw_recs = recommend_products(
                        selected_cust,
                        top_n=max(top_n_count * 4, 20)
                    )
                except Exception as err:
                    st.error(f"Error calling recommendation engine: {err}")
                    raw_recs = pd.DataFrame()

            if raw_recs.empty:
                st.session_state.recommendations = pd.DataFrame()
                st.warning(f"No recommendations available for Customer {selected_cust}.")
            else:
                filtered_recs = raw_recs.copy()

                if sel_cat != "All" and "category" in filtered_recs.columns:
                    filtered_recs = filtered_recs[
                        filtered_recs["category"].astype(str) == sel_cat
                    ]

                if price_limit is not None and "price_usd" in filtered_recs.columns:
                    filtered_recs = filtered_recs[filtered_recs["price_usd"] <= price_limit]

                if "avg_rating" in filtered_recs.columns:
                    filtered_recs = filtered_recs[filtered_recs["avg_rating"] >= min_rating_val]

                filtered_recs = filtered_recs.head(top_n_count).reset_index(drop=True)

                if filtered_recs.empty:
                    st.session_state.recommendations = pd.DataFrame()
                    st.warning("No products matched your specific filter criteria. Try adjusting filters.")
                else:
                    st.session_state.recommendations = filtered_recs

    recs = st.session_state.recommendations

    if not recs.empty:
        st.success(f"Generated {len(recs)} recommendation(s) for Customer `{selected_cust}`")

        # Prepare Downloadable Recommendation Report
        report_df = recs.copy()
        report_df.insert(0, "customer_id", selected_cust)

        rep_cols = [
            c for c in [
                "customer_id",
                "product_id",
                "category",
                "price_usd",
                "avg_rating",
                "recommendation_score",
                "brand_tier",
                "seller_id"
            ] if c in report_df.columns
        ]

        report_csv = report_df[rep_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Recommendation Report (CSV)",
            data=report_csv,
            file_name=f"recommendation_report_{selected_cust}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("🎯 Recommended Products")

        for idx, row in recs.iterrows():
            pid = str(row.get("product_id", "N/A"))
            cat = str(row.get("category", "General"))
            price = row.get("price_usd", "N/A")
            rating = row.get("avg_rating", "N/A")
            score = row.get("recommendation_score", "N/A")

            with st.container(border=True):
                st.markdown(f"### #{idx + 1}. Product ID: `{pid}`")

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.write("🛍️ **Category**")
                    st.write(cat)

                with m2:
                    st.write("💰 **Price**")
                    st.write(f"${price:.2f}" if isinstance(price, (int, float)) else str(price))

                with m3:
                    st.write("⭐ **Rating**")
                    st.write(f"{rating:.2f} / 5.0" if isinstance(rating, (int, float)) else str(rating))

                with m4:
                    st.write("🎯 **Match Score**")
                    st.write(f"{score:.3f}" if isinstance(score, (int, float)) else str(score))

                st.markdown("---")

                # Action Buttons
                b1, b2, b3, b4 = st.columns(4)

                with b1:
                    if st.button("🛒 Add to Cart", key=f"rec_cart_{pid}_{idx}", use_container_width=True):
                        existing_item = next(
                            (x for x in st.session_state.cart if str(x.get("product_id")) == pid),
                            None
                        )
                        if existing_item:
                            existing_item["quantity"] = int(existing_item.get("quantity", 1)) + 1
                            st.success(f"Updated quantity of Product {pid} in cart ({existing_item['quantity']})!")
                        else:
                            item_dict = row.to_dict()
                            item_dict["product_id"] = pid
                            item_dict["quantity"] = 1
                            st.session_state.cart.append(item_dict)
                            st.success(f"Added Product {pid} to cart!")
                        st.rerun()

                with b2:
                    if st.button("❤️ Add to Wishlist", key=f"rec_wish_{pid}_{idx}", use_container_width=True):
                        already_in_wish = any(
                            str(x.get("product_id")) == pid for x in st.session_state.wishlist
                        )
                        if already_in_wish:
                            st.info(f"Product {pid} is already in wishlist.")
                        else:
                            item_dict = row.to_dict()
                            item_dict["product_id"] = pid
                            st.session_state.wishlist.append(item_dict)
                            st.success(f"Added Product {pid} to wishlist!")
                        st.rerun()

                with b3:
                    if st.button("👍 Like", key=f"rec_like_{pid}_{idx}", use_container_width=True):
                        st.session_state.feedback.append({
                            "customer_id": selected_cust,
                            "product_id": pid,
                            "product_name": f"Product {pid}",
                            "feedback": "Like",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success(f"Feedback recorded: Liked Product {pid}")
                        st.rerun()

                with b4:
                    if st.button("👎 Dislike", key=f"rec_dislike_{pid}_{idx}", use_container_width=True):
                        st.session_state.feedback.append({
                            "customer_id": selected_cust,
                            "product_id": pid,
                            "product_name": f"Product {pid}",
                            "feedback": "Dislike",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success(f"Feedback recorded: Disliked Product {pid}")
                        st.rerun()

                # Product Details Expander
                with st.expander("🔎 View Product Details"):
                    prod_info = row.to_dict()
                    seller_id = str(prod_info.get("seller_id", ""))

                    if seller_id and not sellers.empty and "seller_id" in sellers.columns:
                        match_seller = sellers[sellers["seller_id"].astype(str) == seller_id]
                        if not match_seller.empty:
                            s_data = match_seller.iloc[0].to_dict()
                            for k, v in s_data.items():
                                if k not in prod_info:
                                    prod_info[f"seller_{k}"] = v

                    details_table = pd.DataFrame([
                        {"Attribute": str(k), "Value": str(v)}
                        for k, v in prod_info.items()
                        if pd.notna(v)
                    ])

                    st.dataframe(
                        details_table,
                        use_container_width=True,
                        hide_index=True
                    )


# =========================================================
# 5. CART & SIMULATOR
# =========================================================

elif page_key == "Cart":

    st.header("🛒 Shopping Cart & Price Discount Simulator")

    if not st.session_state.cart:
        st.info("Your shopping cart is currently empty. Explore recommendations to add products.")
    else:
        st.write(f"You have **{len(st.session_state.cart)}** unique product type(s) in your cart.")

        gross_total = 0.0
        total_items_count = 0

        for idx, item in enumerate(st.session_state.cart):
            pid = str(item.get("product_id", "N/A"))
            cat = str(item.get("category", "General"))
            price = float(item.get("price_usd", 0.0))
            qty = int(item.get("quantity", 1))
            subtotal = price * qty
            gross_total += subtotal
            total_items_count += qty

            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])

                with c1:
                    st.write(f"📦 **Product ID**: `{pid}`")
                    st.caption(f"Category: {cat}")

                with c2:
                    st.write("💰 **Unit Price**")
                    st.write(f"${price:.2f}")

                with c3:
                    new_qty = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=99,
                        value=qty,
                        key=f"cart_qty_{pid}_{idx}"
                    )
                    if new_qty != qty:
                        st.session_state.cart[idx]["quantity"] = new_qty
                        st.rerun()

                with c4:
                    st.write("💵 **Subtotal**")
                    st.write(f"${subtotal:.2f}")

                with c5:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Remove", key=f"cart_del_{pid}_{idx}", use_container_width=True):
                        st.session_state.cart.pop(idx)
                        st.success(f"Removed Product {pid} from cart.")
                        st.rerun()

        st.markdown("---")
        st.subheader("🏷️ Price & Discount Simulator")

        sim_col1, sim_col2 = st.columns([2, 2])

        with sim_col1:
            st.markdown("##### Apply Promo Code")
            promo_input = st.text_input(
                "Promo Code Input",
                value=st.session_state.applied_promo if st.session_state.applied_promo else "",
                placeholder="SAVE10, SUPER20, STUDENT15, FREESHIP"
            )

            p1, p2 = st.columns(2)
            with p1:
                if st.button("Apply Code", use_container_width=True):
                    clean_code = promo_input.strip().upper()
                    valid_codes = ["SAVE10", "SUPER20", "STUDENT15", "FREESHIP"]
                    if clean_code in valid_codes:
                        st.session_state.applied_promo = clean_code
                        st.success(f"Applied Promo Code `{clean_code}`!")
                        st.rerun()
                    else:
                        st.error("Invalid Code. Try SAVE10, SUPER20, STUDENT15, or FREESHIP.")

            with p2:
                if st.session_state.applied_promo:
                    if st.button("Remove Code", use_container_width=True):
                        st.session_state.applied_promo = None
                        st.info("Promo code removed.")
                        st.rerun()

            st.caption("💡 Test Promo Codes: `SAVE10` (10% off), `SUPER20` (20% off), `STUDENT15` (15% off), `FREESHIP` ($0 Shipping)")

        # Discount & Shipping Calculations
        promo_pct = 0.0
        free_shipping = False

        if st.session_state.applied_promo == "SAVE10":
            promo_pct = 0.10
        elif st.session_state.applied_promo == "SUPER20":
            promo_pct = 0.20
        elif st.session_state.applied_promo == "STUDENT15":
            promo_pct = 0.15
        elif st.session_state.applied_promo == "FREESHIP":
            free_shipping = True

        promo_savings = gross_total * promo_pct

        bulk_pct = 0.0
        if total_items_count >= 5:
            bulk_pct = 0.10
        elif total_items_count >= 3:
            bulk_pct = 0.05

        bulk_savings = gross_total * bulk_pct

        std_shipping = 5.00
        shipping_cost = 0.0 if (free_shipping or gross_total == 0) else std_shipping

        net_total = max(gross_total - promo_savings - bulk_savings + shipping_cost, 0.0)
        total_savings = promo_savings + bulk_savings + (std_shipping if free_shipping else 0.0)

        with sim_col2:
            with st.container(border=True):
                st.markdown("#### 🧾 Order Summary & Receipt")
                st.write(f"Gross Subtotal ({total_items_count} items): **${gross_total:.2f}**")

                if promo_savings > 0:
                    st.write(f"🏷️ Promo Discount ({int(promo_pct*100)}%): <span style='color: #ef4444;'>-${promo_savings:.2f}</span>", unsafe_allow_html=True)

                if bulk_savings > 0:
                    st.write(f"📦 Bulk Discount ({int(bulk_pct*100)}%): <span style='color: #ef4444;'>-${bulk_savings:.2f}</span>", unsafe_allow_html=True)

                if free_shipping:
                    st.write("🚚 Shipping Fee: <span style='color: #22c55e;'>FREE ($0.00)</span>", unsafe_allow_html=True)
                else:
                    st.write(f"🚚 Shipping Fee: **${shipping_cost:.2f}**")

                st.markdown("---")
                st.markdown(f"### Net Total: <span style='color: #ef4444;'>${net_total:.2f}</span>", unsafe_allow_html=True)

                if total_savings > 0:
                    st.caption(f"🎉 Total Savings Realized: **${total_savings:.2f}**")

        st.markdown("---")

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("🗑️ Clear Cart", use_container_width=True):
                st.session_state.cart = []
                st.session_state.applied_promo = None
                st.success("Cart cleared!")
                st.rerun()

        with col_b2:
            if st.button("💳 Proceed to Checkout (DEMO ONLY)", use_container_width=True):
                st.balloons()
                st.success(
                    f"🎉 **DEMO CHECKOUT COMPLETED SUCCESSFULLY!**\n\n"
                    f"Order Summary:\n"
                    f"- Total Items: **{total_items_count}**\n"
                    f"- Amount Charged: **${net_total:.2f}**\n"
                    f"- Total Savings: **${total_savings:.2f}**\n\n"
                    f"*Note: This is a DEMO CHECKOUT for testing. No real payments were processed.*"
                )


# =========================================================
# 6. WISHLIST
# =========================================================

elif page_key == "Wishlist":

    st.header("❤️ Saved Wishlist")

    if not st.session_state.wishlist:
        st.info("Your wishlist is currently empty. Add products to your wishlist from Recommendations.")
    else:
        st.write(f"You have **{len(st.session_state.wishlist)}** saved item(s).")

        for idx, item in enumerate(st.session_state.wishlist):
            pid = str(item.get("product_id", "N/A"))
            cat = str(item.get("category", "General"))
            price = float(item.get("price_usd", 0.0))
            rating = item.get("avg_rating", "N/A")

            with st.container(border=True):
                w1, w2, w3, w4, w5 = st.columns([3, 2, 2, 2, 2])

                with w1:
                    st.write(f"❤️ **Product ID**: `{pid}`")
                    st.caption(f"Category: {cat}")

                with w2:
                    st.write("💰 **Price**")
                    st.write(f"${price:.2f}")

                with w3:
                    st.write("⭐ **Rating**")
                    st.write(f"{rating:.2f} / 5.0" if isinstance(rating, (int, float)) else str(rating))

                with w4:
                    if st.button("🛒 Move to Cart", key=f"wish_mv_{pid}_{idx}", use_container_width=True):
                        existing_item = next(
                            (c for c in st.session_state.cart if str(c.get("product_id")) == pid),
                            None
                        )
                        if existing_item:
                            existing_item["quantity"] = int(existing_item.get("quantity", 1)) + 1
                        else:
                            cart_item = item.copy()
                            cart_item["quantity"] = 1
                            st.session_state.cart.append(cart_item)

                        st.session_state.wishlist.pop(idx)
                        st.success(f"Moved Product {pid} to cart!")
                        st.rerun()

                with w5:
                    if st.button("🗑️ Remove", key=f"wish_del_{pid}_{idx}", use_container_width=True):
                        st.session_state.wishlist.pop(idx)
                        st.success(f"Removed Product {pid} from wishlist.")
                        st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Wishlist", use_container_width=True):
            st.session_state.wishlist = []
            st.success("Wishlist cleared!")
            st.rerun()


# =========================================================
# 7. FEEDBACK
# =========================================================

elif page_key == "Feedback":

    st.header("💬 Customer Feedback Log")

    if not st.session_state.feedback:
        st.info("No feedback recorded yet in this session. Give feedback using 👍 Like or 👎 Dislike buttons on the Recommendations page.")
    else:
        fb_df = pd.DataFrame(st.session_state.feedback)

        # Feedback Metrics
        total_fb = len(fb_df)
        likes_count = len(fb_df[fb_df["feedback"] == "Like"])
        dislikes_count = len(fb_df[fb_df["feedback"] == "Dislike"])
        sat_pct = (likes_count / total_fb * 100) if total_fb > 0 else 0.0

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.metric("💬 Total Feedback Entries", total_fb)

        with f2:
            st.metric("👍 Likes", likes_count)

        with f3:
            st.metric("👎 Dislikes", dislikes_count)

        with f4:
            st.metric("📊 Satisfaction Ratio", f"{sat_pct:.1f}%")

        st.markdown("---")
        st.subheader("📜 Recorded Feedback Records")

        st.dataframe(fb_df, use_container_width=True)

        col_fb1, col_fb2 = st.columns(2)

        with col_fb1:
            fb_csv_data = fb_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Feedback CSV",
                data=fb_csv_data,
                file_name="recommendation_feedback_log.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_fb2:
            if st.button("🗑️ Clear Feedback Log", use_container_width=True):
                st.session_state.feedback = []
                st.success("Feedback log cleared!")
                st.rerun()