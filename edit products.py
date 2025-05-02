import streamlit as st
import pandas as pd
import os

# Load the existing products dataset
def load_products(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['Product ID', 'Category', 'Rating', 'Users Purchased'])

# Save the updated products dataset
def save_products(products_df, file_path):
    products_df.to_csv(file_path, index=False)

# Generate a new product ID
def generate_product_id(existing_ids):
    if len(existing_ids) == 0:
        return 'P0001'
    else:
        last_id = max([int(pid[1:]) for pid in existing_ids])
        new_id = f'P{last_id + 1:04d}'
        return new_id

# File path
file_path = 'products_dataset.csv'

# Streamlit app
st.set_page_config(page_title="Manage Products", page_icon="📅", layout="centered")

st.title('🔢 Manage Products in Dataset')

# Load existing data
products_df = load_products(file_path)

# Form to add a new product
st.subheader('Add New Product')
with st.form(key='add_product_form'):
    category = st.selectbox('Select Product Category:', ['Shoes', 'Watches', 'Bags', 'Sunglasses', 'Smartphones'])
    rating = st.slider('Product Rating:', min_value=1.0, max_value=5.0, step=0.1)
    users_purchased = st.number_input('Number of Users Purchased:', min_value=0, step=1)
    submit_button = st.form_submit_button(label='Add Product')

    if submit_button:
        new_product_id = generate_product_id(products_df['Product ID'].tolist())
        new_product = pd.DataFrame({
            'Product ID': [new_product_id],
            'Category': [category],
            'Rating': [rating],
            'Users Purchased': [int(users_purchased)]
        })
        
        products_df = pd.concat([products_df, new_product], ignore_index=True)
        save_products(products_df, file_path)
        st.success(f'Product {new_product_id} added successfully!')

# Form to delete a product
st.subheader('Delete Product')
with st.form(key='delete_product_form'):
    product_to_delete = st.selectbox('Select Product ID to Delete:', products_df['Product ID'].unique())
    delete_button = st.form_submit_button(label='Delete Product')

    if delete_button:
        products_df = products_df[products_df['Product ID'] != product_to_delete]
        save_products(products_df, file_path)
        st.success(f'Product {product_to_delete} deleted successfully!')

# Show the updated product list
st.markdown('---')
st.subheader('Current Products in Dataset')
st.dataframe(products_df, use_container_width=True)
