import pandas as pd
import json
import gzip
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import ndcg_score
from datetime import datetime

# record current timestamp
start = datetime.now()

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
sample_size = 500000  # Adjust the sample size according to your computational capacity
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

# Convert user_id and asin to integer indices
user_ids = pd.factorize(train_data['user_id'])[0]
item_ids = pd.factorize(train_data['asin'])[0]

# Create user-item matrix using sparse representation
user_item_matrix = csr_matrix((train_data['rating'], (user_ids, item_ids)))

# Create a NearestNeighbors model
k = 10  # Number of similar users to consider
model = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
model.fit(user_item_matrix)

# Function to get top-k similar users
def get_top_similar_users(user_id, k=10):
    user_indices = user_ids[train_data['user_id'] == user_id]
    if len(user_indices) == 0:
        return []  # Return an empty list if the user ID is not found
    similar_users = []
    for user_index in user_indices:
        _, indices = model.kneighbors(user_item_matrix[user_index], n_neighbors=k+1)
        similar_users.extend(indices.flatten()[1:])  # Exclude the user itself
    similar_users = np.unique(similar_users)
    return train_data['user_id'].unique()[similar_users]

# Function to recommend items for a user
def recommend_items(user_id, top_n=10):
    # Get top similar users
    similar_users = get_top_similar_users(user_id)
    
    if len(similar_users) == 0:
        return []  # Return an empty list if no similar users are found
    
    # Get the items rated by similar users
    similar_user_items = train_data[train_data['user_id'].isin(similar_users)]
    
    # Calculate the weighted average rating for each item
    item_scores = similar_user_items.groupby('asin')['rating'].sum() / similar_user_items.groupby('asin')['user_id'].count()
    
    # Remove items already rated by the user
    user_rated_items = train_data[train_data['user_id'] == user_id]['asin'].unique()
    item_scores = item_scores[~item_scores.index.isin(user_rated_items)]
    
    # Sort items by score and get top-n recommendations
    top_items = item_scores.sort_values(ascending=False)[:top_n]
    
    # Get the product titles for the recommended items
    recommended_titles = train_data[train_data['asin'].isin(top_items.index)][['asin', 'title']]
    
    return recommended_titles.values.tolist()  # Return a list of [asin, title] pairs

# Evaluation metrics
def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    precision = len(actual_set & predicted_set) / k
    return precision

def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    recall = len(actual_set & predicted_set) / len(actual_set)
    return recall

# Evaluate the model
def evaluate_model(test_data, top_n=10):
    user_items = test_data.groupby('user_id')['asin'].apply(list).to_dict()
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id, actual_items in user_items.items():
        recommended_items = recommend_items(user_id, top_n)
        recommended_asins = [item[0] for item in recommended_items]  # Extract ASINs from recommended items
        precision = precision_at_k(actual_items, recommended_asins, top_n)
        recall = recall_at_k(actual_items, recommended_asins, top_n)

        # Calculate NDCG score only if there are more than one actual item
        ndcg = 0
        if len(actual_items) > 1:
            relevance_scores = [1 if item in actual_items else 0 for item in recommended_asins]
            ideal_relevance_scores = sorted(relevance_scores, reverse=True)
            if len(relevance_scores) > 0:
                ndcg = ndcg_score([relevance_scores], [ideal_relevance_scores])

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)

    print(f"Precision@{top_n}: {avg_precision:.4f}")
    print(f"Recall@{top_n}: {avg_recall:.4f}")
    print(f"NDCG@{top_n}: {avg_ndcg:.4f}")

evaluate_model(test_data)

# Randomly select a user ID from the sample data
sample_user_id = df['user_id'].sample(1).iloc[0]

# Generate recommendations for the selected user
recommended_items = recommend_items(sample_user_id)
print(f"Recommended items for user {sample_user_id}:")
for item in recommended_items:
    print(f"ASIN: {item[0]}, Title: {item[1]}")
    
# record end timestamp
end = datetime.now()

# find difference loop start and end time and display
td = (end - start).total_seconds() * 10**3
print(f"The time of execution of above program is : {td:.03f}ms")