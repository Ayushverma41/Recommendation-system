import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-Commerce Product Recommender", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("products_dataset.csv")

df = load_data()

# Read the product_id from the URL

product_id = st.query_params.get("product_id", None)

if product_id:
    st.title("📦 Product Details")
    product_info = df[df['Product ID'] == product_id]
    
    if not product_info.empty:
        st.subheader(f"Details for Product ID: {product_id}")
        st.table(product_info.T)  # Display as transposed table
    else:
        st.warning(f"No product found with ID '{product_id}'")
else:
    st.title("Welcome to Product Viewer")
    st.info("No product selected. Please open this page via a product link from the recommendation system.")

# Sidebar for manual selection (optional navigation within app02)
st.sidebar.header("🔍 Browse Products")
n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)


# KNN-based recommendations
features = df[["Users Purchased", "Rating", "Price"]]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

knn_model = NearestNeighbors(metric='euclidean')
knn_model.fit(features_scaled)

def recommend_products(product_id, n_neighbors=5):
    try:
        index = df[df["Product ID"] == product_id].index[0]
        category = df.at[index, "Category"]
        vector = features_scaled[index].reshape(1, -1)
        distances, indices = knn_model.kneighbors(vector, n_neighbors=len(df))
        recommended_indices = [i for i in indices.flatten() if i != index and df.at[i, "Category"] == category]
        top_indices = recommended_indices[:n_neighbors]
        return df.iloc[top_indices][["Product ID", "Category", "Rating", "Users Purchased", "Price"]]
    except:
        return pd.DataFrame()

def category_recommendation(product_id):
    category_map = {
        "Watches": ["Sunglasses", "Men Wallet"],
        "Bags": ["Shoes"],
        "Smartphones": ["Earbuds"],
        "Shoes": ["Bags"],
        "Sunglasses": ["Watches"],
        "Men Wallet": ["Watches"],
        "Earbuds": ["Smartphones"]
    }
    try:
        category = df[df["Product ID"] == product_id]["Category"].values[0]
        related_categories = category_map.get(category, [])
        if not related_categories:
            return pd.DataFrame()
        filtered = df[df["Category"].isin(related_categories)]
        return filtered.sort_values(by=["Rating", "Users Purchased"], ascending=False).head(10)[
            ["Product ID", "Category", "Rating", "Users Purchased", "Price"]
        ]
    except:
        return pd.DataFrame()

# Show recommendations only if product_id is valid
if product_id and product_id in df["Product ID"].values:
    st.subheader("🤖 Similar Products (KNN Recommendations)")
    st.dataframe(recommend_products(product_id, n_recommendations), use_container_width=True)

    st.subheader("🎯 Complementary Product Recommendations")
    comp = category_recommendation(product_id)
    if not comp.empty:
        st.dataframe(comp, use_container_width=True)
    else:
        st.info("No complementary category mapping found for this product.")
