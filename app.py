import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# -------------------------
# GLOBAL CONFIG
# -------------------------
st.set_page_config(page_title="Unified Product App", layout="wide")
FILE_PATH = 'products_dataset.csv'

# -------------------------
# BEAUTIFYING CSS
# -------------------------
st.markdown("""
<style>
/* Adapt background & text color to current theme */
.stApp {
    background: var(--background-color);
    font-family: 'Segoe UI', sans-serif;
    color: var(--text-color);
}

/* Theme-aware titles */
h1, h2, h3 {
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--sidebar-background-color);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
section[data-testid="stSidebar"] * {
    color: var(--text-color) !important;
}

/* Navigation Radio Buttons */
[data-testid="stSidebar"] .stRadio > label {
    font-weight: bold;
    font-size: 1rem;
    color: var(--primary-color);
}
[data-testid="stSidebar"] .stRadio div[role='radiogroup'] label {
    background-color: var(--sidebar-option-bg);
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.4rem;
    transition: all 0.2s ease-in-out;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio div[role='radiogroup'] label:hover {
    background-color: var(--sidebar-option-hover-bg);
    transform: scale(1.02);
}
[data-testid="stSidebar"] .stRadio div[role='radiogroup'] label[data-selected="true"] {
    background-color: var(--primary-color);
    color: white !important;
    font-weight: 600;
}

/* Dataframe Table */
table {
    border-radius: 10px;
    overflow: hidden;
}
thead th {
    background-color: var(--primary-color) !important;
    color: white !important;
    text-align: center;
}
td, th {
    padding: 10px !important;
    text-align: center;
}

/* Links */
a {
    display: inline-block;
    padding: 6px 12px;
    background: var(--primary-color);
    color: white !important;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 500;
}
a:hover {
    background: var(--secondary-color);
    color: black !important;
}

/* Alerts */
.stAlert {
    background-color: var(--alert-bg);
    border: 1px solid var(--alert-border);
    color: var(--text-color);
}

/* Define theme-aware variables */
:root {
    --primary-color: #007cf0;
    --secondary-color: #00dfd8;
    --background-color: #ffffff;
    --text-color: #000000;
    --sidebar-background-color: rgba(255,255,255,0.95);
    --sidebar-option-bg: rgba(0,0,0,0.05);
    --sidebar-option-hover-bg: rgba(0,0,0,0.1);
    --alert-bg: rgba(0,0,0,0.03);
    --alert-border: rgba(0,0,0,0.1);
}

@media (prefers-color-scheme: dark) {
    :root {
        --primary-color: #00dfd8;
        --secondary-color: #007cf0;
        --background-color: #0f2027;
        --text-color: #e0e0e0;
        --sidebar-background-color: rgba(30,30,30,0.95);
        --sidebar-option-bg: rgba(255,255,255,0.05);
        --sidebar-option-hover-bg: rgba(255,255,255,0.1);
        --alert-bg: rgba(255,255,255,0.05);
        --alert-border: rgba(255,255,255,0.15);
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# DATA HELPERS
# -------------------------
@st.cache_data
def load_products():
    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)
    else:
        return pd.DataFrame(columns=['Product ID', 'Category', 'Rating', 'Users Purchased', 'Price'])

def save_products(df):
    df.to_csv(FILE_PATH, index=False)

# -------------------------
# FUNCTIONS
# -------------------------
def generate_interactions(products, num_users=500):
    if products.empty:
        return pd.DataFrame(columns=['UserID', 'ProductID', 'Purchased'])
    user_ids = [f'U{i:03d}' for i in range(1, num_users + 1)]
    interactions = []
    for user in user_ids:
        sample_size = min(len(products), random.randint(1, 5))
        purchased_products = random.sample(list(products['Product ID']), k=sample_size)
        for product in purchased_products:
            interactions.append((user, product, 1))
    return pd.DataFrame(interactions, columns=['UserID', 'ProductID', 'Purchased'])

def recommend_products_by_category(products, interactions_df, category, top_n=5):
    if products.empty or interactions_df.empty:
        return pd.DataFrame()
    category_products = products[products['Category'] == category]
    product_popularity = interactions_df[interactions_df['ProductID'].isin(category_products['Product ID'])] \
                         .groupby('ProductID').size().sort_values(ascending=False)
    top_products = product_popularity.index.tolist()
    recommended = products[products['Product ID'].isin(top_products)]
    recommended = recommended[recommended['Category'] == category]
    recommended = recommended.sort_values(by='Rating', ascending=False).head(top_n)
    return recommended[['Product ID', 'Category', 'Rating', 'Price']]

def knn_recommend(df, product_id, n_neighbors=5):
    if df.empty:
        return pd.DataFrame()
    features = df[["Users Purchased", "Rating", "Price"]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    knn_model = NearestNeighbors(metric='euclidean')
    knn_model.fit(features_scaled)
    try:
        index = df[df["Product ID"] == product_id].index[0]
        category = df.at[index, "Category"]
        vector = features_scaled[index].reshape(1, -1)
        distances, indices = knn_model.kneighbors(vector, n_neighbors=len(df))
        rec_indices = [i for i in indices.flatten() if i != index and df.at[i, "Category"] == category]
        return df.iloc[rec_indices[:n_neighbors]][["Product ID", "Category", "Rating", "Users Purchased", "Price"]]
    except:
        return pd.DataFrame()

def complementary_recommend(df, product_id):
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
        related = category_map.get(category, [])
        if not related:
            return pd.DataFrame()
        filtered = df[df["Category"].isin(related)]
        return filtered.sort_values(by=["Rating", "Users Purchased"], ascending=False).head(10)[
            ["Product ID", "Category", "Rating", "Users Purchased", "Price"]
        ]
    except:
        return pd.DataFrame()

def generate_product_id(existing_ids):
    if not existing_ids:
        return 'P0001'
    last_id = max([int(pid[1:]) for pid in existing_ids])
    return f'P{last_id + 1:04d}'

# -------------------------
# QUERY PARAM LOGIC
# -------------------------
query_params = st.query_params if hasattr(st, "query_params") else {}
selected_page = query_params.get("page", "🏷 Product Recommendations")
selected_product = query_params.get("product_id", "")

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏷 Product Recommendations", "📦 Product Details", "🛠 Manage Products"],
                        index=["🏷 Product Recommendations", "📦 Product Details", "🛠 Manage Products"].index(selected_page))

products_df = load_products()

# -------------------------
# PAGE 1: RECOMMENDATIONS
# -------------------------
if page == "🏷 Product Recommendations":
    st.title("🛒 Product Recommendation System")
    if products_df.empty:
        st.warning("No products available. Please add products in the 'Manage Products' section.")
    else:
        st.dataframe(products_df.head(20), use_container_width=True)
        interactions_df = generate_interactions(products_df)
        category = st.sidebar.selectbox('Select Category', products_df['Category'].unique())
        num_products = st.sidebar.slider('Number of Products', 1, 20, 5)

        if st.sidebar.button('Recommend'):
            recs = recommend_products_by_category(products_df, interactions_df, category, num_products)
            if not recs.empty:
                recs["Product ID"] = recs["Product ID"].apply(lambda pid: f'<a href="?page=📦+Product+Details&product_id={pid}">{pid}</a>')
                st.success(f"Top {num_products} products in '{category}'")
                st.write(recs.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No recommendations available.")

# -------------------------
# PAGE 2: PRODUCT DETAILS
# -------------------------
elif page == "📦 Product Details":
    st.title("📦 Product Viewer & Recommendations")
    product_id = st.text_input("Enter Product ID", value=selected_product)

    if product_id:
        info = products_df[products_df['Product ID'] == product_id]
        if not info.empty:
            st.subheader("Product Info")
            st.table(info.T)
            st.subheader("🤖 Similar Products")
            st.dataframe(knn_recommend(products_df, product_id), use_container_width=True)
            st.subheader("🎯 Complementary Products")
            comp = complementary_recommend(products_df, product_id)
            if not comp.empty:
                st.dataframe(comp, use_container_width=True)
            else:
                st.info("No complementary category mapping found.")
        else:
            st.warning("Product not found.")

# -------------------------
# PAGE 3: MANAGE PRODUCTS
# -------------------------
elif page == "🛠 Manage Products":
    st.title("Manage Product Dataset")

    # Add Product
    st.subheader("Add New Product")
    with st.form(key='add_product_form'):
        category = st.selectbox('Category', ['Shoes', 'Watches', 'Bags', 'Sunglasses', 'Smartphones'])
        rating = st.slider('Rating', 1.0, 5.0, 4.0, 0.1)
        users_purchased = st.number_input('Users Purchased', min_value=0, step=1)
        price = st.number_input('Price ($)', min_value=0.0, step=0.01)
        submit = st.form_submit_button("Add Product")
        if submit:
            new_id = generate_product_id(products_df['Product ID'].tolist())
            new_product = pd.DataFrame({
                'Product ID': [new_id],
                'Category': [category],
                'Rating': [rating],
                'Users Purchased': [int(users_purchased)],
                'Price': [float(price)]
            })
            products_df = pd.concat([products_df, new_product], ignore_index=True)
            save_products(products_df)
            st.success(f"Product {new_id} added!")

    # Delete Product
    st.subheader("Delete Product")
    if not products_df.empty:
        with st.form(key='delete_product_form'):
            prod_to_delete = st.selectbox('Select Product', products_df['Product ID'].unique())
            delete_btn = st.form_submit_button("Delete")
            if delete_btn:
                products_df = products_df[products_df['Product ID'] != prod_to_delete]
                save_products(products_df)
                st.success(f"Product {prod_to_delete} deleted!")

    # Show Products
    st.subheader("Current Products")
    st.dataframe(products_df, use_container_width=True)
