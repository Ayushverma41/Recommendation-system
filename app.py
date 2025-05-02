import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Product Recommendation System", page_icon="🛒", layout="wide")

# Load the products dataset
def load_products():
    products = pd.read_csv('products_dataset.csv')
    return products

# Simulate user-product interactions
def generate_interactions(products, num_users=500):
    user_ids = [f'U{i:03d}' for i in range(1, num_users + 1)]
    interactions = []
    for user in user_ids:
        purchased_products = random.sample(list(products['Product ID']), k=random.randint(5, 20))
        for product in purchased_products:
            interactions.append((user, product, 1))  # 1 means purchased
    interactions_df = pd.DataFrame(interactions, columns=['UserID', 'ProductID', 'Purchased'])
    return interactions_df

# Create user-item matrix
def create_user_item_matrix(interactions_df):
    user_item_matrix = interactions_df.pivot_table(index='UserID', columns='ProductID', values='Purchased', fill_value=0)
    return user_item_matrix

# Build item similarity matrix
def build_item_similarity_matrix(user_item_matrix):
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return item_similarity_df

# Recommend products based on category
def recommend_products_by_category(products, interactions_df, category, top_n=5):
    category_products = products[products['Category'] == category]
    product_popularity = interactions_df[interactions_df['ProductID'].isin(category_products['Product ID'])]\
                         .groupby('ProductID').size().sort_values(ascending=False)
    top_products = product_popularity.index.tolist()
    recommended = products[products['Product ID'].isin(top_products)]
    recommended = recommended[recommended['Category'] == category]
    recommended = recommended.sort_values(by='Rating', ascending=False).head(top_n)
    return recommended[['Product ID', 'Category', 'Rating']]

# Apply background and styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f2f6, #c9d6ff);
        font-family: 'Arial', sans-serif;
    }
    .big-font {
        font-size:40px !important;
        color: #1f77b4;
        text-align: center;
    }
    .small-font {
        font-size:18px !important;
        color: #4f4f4f;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="big-font">🛒 Product Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Choose a category and discover top-rated products!</p>', unsafe_allow_html=True)
st.markdown("---")

# Load data
products = load_products()
interactions_df = generate_interactions(products)
user_item_matrix = create_user_item_matrix(interactions_df)
item_similarity_df = build_item_similarity_matrix(user_item_matrix)

# Sidebar for user input
st.sidebar.header('🔎 Customize Your Search')
category = st.sidebar.selectbox('Select a Product Category:', products['Category'].unique())
num_products = st.sidebar.slider('Number of Products to Recommend:', min_value=1, max_value=20, value=5)

# Recommend button
if st.sidebar.button('🔔 Recommend'):
    with st.spinner('Generating your recommendations... 🚀'):
        recommendations = recommend_products_by_category(products, interactions_df, category, num_products)
        
        st.success(f"Here are the Top {num_products} '{category}' products for you!")
        
        st.dataframe(
            recommendations.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )

st.markdown("---")
st.markdown("<center>Its a team ❤️ effort</center>", unsafe_allow_html=True)
