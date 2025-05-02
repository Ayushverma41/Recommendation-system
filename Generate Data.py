import csv
import random

# Define the categories
categories = ['Shoes', 'Watches', 'Bags', 'Sunglasses', 'Smartphones']

# Prepare the dataset
products = []
for i in range(1, 1001):
    product_id = f'P{i:04d}'  # e.g., P0001, P0002, etc.
    category = random.choice(categories)
    rating = round(random.uniform(1.0, 5.0), 1)  # Rating between 1.0 and 5.0
    users_purchased = random.randint(1, 10000)  # Random number between 1 and 10000
    products.append([product_id, category, rating, users_purchased])

# Write to CSV
csv_file = '/mnt/data/products_dataset.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Product ID', 'Category', 'Rating', 'Users Purchased'])  # Header
    writer.writerows(products)

csv_file
