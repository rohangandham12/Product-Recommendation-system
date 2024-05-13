import pandas as pd
import json
import gzip
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import ndcg_score

def load_data(file_path, sample_size=None):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size is not None and i >= sample_size:
                break
            review = json.loads(line.strip())
            data.append({
                'user_id': review['user_id'],
                'asin': review['asin'],
                'rating': review['rating'],
                'title': review['title']  # Add the 'title' field
            })
    return pd.DataFrame(data)

# Load a sample of the data
file_path = 'All_Beauty.jsonl.gz'
sample_size = 1000  # Adjust the sample size according to your computational capacity
df = load_data(file_path, sample_size)

# Filter out users and items with few interactions
min_user_interactions = 3
min_item_interactions = 3
user_counts = df['user_id'].value_counts()
item_counts = df['asin'].value_counts()
filtered_users = user_counts[user_counts >= min_user_interactions].index
filtered_items = item_counts[item_counts >= min_item_interactions].index
df = df[df['user_id'].isin(filtered_users) & df['asin'].isin(filtered_items)]

# Check if the filtered dataset is empty
if df.empty:
    raise ValueError("The filtered dataset is empty. Adjust the filtering thresholds or increase the sample size.")

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Create a mapping dictionary for item IDs
item_id_map = {item_id: idx for idx, item_id in enumerate(train_data['asin'].unique())}

# Convert user_id and asin to integer indices
user_ids = pd.factorize(train_data['user_id'])[0]
item_ids = train_data['asin'].map(item_id_map).astype(int)

# Create user-item matrix using sparse representation
user_item_matrix = csr_matrix((train_data['rating'], (user_ids, item_ids)))

# Create a NearestNeighbors model for item-based similarity
model_item_based = NearestNeighbors(metric='cosine', algorithm='brute')
model_item_based.fit(user_item_matrix.T)

# Function to recommend items based on item similarity
# Function to recommend items based on item similarity
def recommend_items_item_based(item_id, model, train_data, item_id_map, user_item_matrix, top_n=10):
    if item_id not in item_id_map:
        print(f"Item {item_id} not found in the training data.")
        return []
    
    item_index = item_id_map[item_id]
    item_vector = user_item_matrix.T[item_index]
    n_samples = user_item_matrix.T.shape[0]
    n_neighbors = min(top_n + 1, n_samples)  # Ensure n_neighbors doesn't exceed the available samples
    distances, indices = model.kneighbors(item_vector.toarray().reshape(1, -1), n_neighbors=n_neighbors)
    similar_item_indices = indices.flatten()[1:]  # Exclude the item itself
    similar_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(idx)] for idx in similar_item_indices]
    recommended_items = train_data[train_data['asin'].isin(similar_item_ids)][['asin', 'title']]
    return recommended_items.values.tolist()

# Evaluation metrics
def evaluate_model(test_data, train_data, model, item_id_map, user_item_matrix, top_n=10):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    # Get unique item IDs from the test data
    test_item_ids = test_data['asin'].unique()

    for item_id in test_item_ids:
        if item_id not in item_id_map:
            continue  # Skip evaluation for items not in the training data
        
        recommended_items = recommend_items_item_based(item_id, model, train_data, item_id_map, user_item_matrix, top_n)
        recommended_asins = [item[0] for item in recommended_items]  # Extract ASINs from recommended items
        
        # Get actual users who interacted with the item in the test data
        actual_users = test_data[test_data['asin'] == item_id]['user_id'].unique()
        
        # Calculate precision
        precision = len(set(recommended_asins) & set(train_data[train_data['user_id'].isin(actual_users)]['asin'].unique())) / top_n
        
        # Calculate recall
        recall = len(set(recommended_asins) & set(train_data[train_data['user_id'].isin(actual_users)]['asin'].unique())) / len(actual_users)
        
        # Calculate NDCG
        relevance_scores = [1 if asin in train_data[train_data['user_id'].isin(actual_users)]['asin'].unique() else 0 for asin in recommended_asins]
        ndcg = ndcg_score([relevance_scores], [np.ones(len(relevance_scores))])
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)

    print(f"Precision@{top_n}: {avg_precision:.4f}")
    print(f"Recall@{top_n}: {avg_recall:.4f}")
    print(f"NDCG@{top_n}: {avg_ndcg:.4f}")

evaluate_model(test_data, train_data, model_item_based, item_id_map, user_item_matrix)

# Example usage
sample_item_id = train_data['asin'].sample(1).iloc[0]  # Sample an item ID from the training data
recommended_items = recommend_items_item_based(sample_item_id, model_item_based, train_data, item_id_map, user_item_matrix)
if recommended_items:
    print(f"Recommended items similar to {sample_item_id}:")
    for item in recommended_items:
        print(f"ASIN: {item[0]}, Title: {item[1]}")
else:
    print(f"No recommendations available for item {sample_item_id}.")
