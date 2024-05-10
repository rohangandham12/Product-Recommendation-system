import gzip
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load and parse the meta_All_Beauty.jsonl.gz file
def load_meta_data(file_path):
    meta_data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            meta_data.append(json.loads(line.strip()))
    return meta_data

# Load and parse the All_Beauty.jsonl.gz file
def load_review_data(file_path):
    review_data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            review_data.append(json.loads(line.strip()))
    return review_data

# Create user-item sparse matrix
def create_user_item_matrix(review_data):
    user_item_dict = {}
    for review in review_data:
        user_id = review.get('user_id', review.get('reviewerID'))
        item_id = review['asin']
        if user_id not in user_item_dict:
            user_item_dict[user_id] = {}
        user_item_dict[user_id][item_id] = 1

    users = list(user_item_dict.keys())
    items = set(item for user_items in user_item_dict.values() for item in user_items.keys())
    user_item_matrix = np.zeros((len(users), len(items)), dtype=np.int32)

    for user_idx, user in enumerate(users):
        for item_idx, item in enumerate(items):
            if item in user_item_dict[user]:
                user_item_matrix[user_idx, item_idx] = 1

    return user_item_matrix, users, items

# Get item popularity scores
def get_item_popularity(user_item_matrix):
    item_popularity = user_item_matrix.sum(axis=0)
    item_popularity = (item_popularity - item_popularity.min()) / (item_popularity.max() - item_popularity.min())
    return item_popularity

# Get user-based recommendations
def get_user_based_recommendations(user_item_matrix, user_idx, item_popularity, topn=5, popularity_weight=0.5):
    user_similarities = cosine_similarity(user_item_matrix)[user_idx]
    similar_users = user_similarities.argsort()[::-1]

    item_scores = {}
    for other_user in similar_users:
        if other_user != user_idx:
            other_user_items = user_item_matrix[other_user].nonzero()[0]
            for item_idx in other_user_items:
                item_score = user_similarities[other_user] + (popularity_weight * item_popularity[item_idx])
                item_scores[item_idx] = item_scores.get(item_idx, 0) + item_score

    sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_items = [item_idx for item_idx, _ in sorted_item_scores[:topn]]

    return recommended_items

def precision_at_k(recommended_items, ground_truth_items, k):
    if k == 0:
        return 0.0
    recommended_items = set(recommended_items[:k])
    ground_truth_items = set(ground_truth_items)
    precision = len(recommended_items & ground_truth_items) / k
    return precision

def recall_at_k(recommended_items, ground_truth_items, k):
    if k == 0 or len(ground_truth_items) == 0:
        return 0.0
    recommended_items = set(recommended_items[:k])
    ground_truth_items = set(ground_truth_items)
    recall = len(recommended_items & ground_truth_items) / len(ground_truth_items)
    return recall

def ndcg_at_k(recommended_items, ground_truth_items, k):
    if k == 0:
        return 0.0
    
    recommended_items = recommended_items[:k]
    relevance_scores = [1 if item in ground_truth_items else 0 for item in recommended_items]
    
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    dcg = np.sum(relevance_scores / discounts)
    
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = np.sum(ideal_relevance_scores / discounts)
    
    if ideal_dcg == 0:
        return 0.0
    
    ndcg = dcg / ideal_dcg
    return ndcg

# Evaluate the recommender model
def evaluate_recommender(user_item_matrix, users, items, topn=5, popularity_weight=0.5):
    item_popularity = get_item_popularity(user_item_matrix)

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_idx, user in enumerate(users):
        user_items = user_item_matrix[user_idx].nonzero()[0]
        recommended_items = get_user_based_recommendations(user_item_matrix, user_idx, item_popularity, topn, popularity_weight)

        precision = precision_at_k(recommended_items, user_items, topn)
        recall = recall_at_k(recommended_items, user_items, topn)
        ndcg = ndcg_at_k(recommended_items, user_items, topn)

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)

    return avg_precision, avg_recall, avg_ndcg

# Example usage
meta_file_path = 'meta_All_Beauty.jsonl.gz'
review_file_path = 'All_Beauty.jsonl.gz'

try:
    meta_data = load_meta_data(meta_file_path)
    review_data = load_review_data(review_file_path)
except FileNotFoundError:
    print("Error: File not found. Please check the file paths.")
    exit(1)
except gzip.BadGzipFile:
    print("Error: Invalid gzip file. Please check the file integrity.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format. Please check the file content.")
    exit(1)

# Sample a subset of the review data
sample_size = 800  # Adjust this value to control the sample size
sample_data = random.sample(review_data, sample_size)

user_item_matrix, users, items = create_user_item_matrix(sample_data)

num_tests = 5
topn = 5
popularity_weight = 0.5  # Adjust this value to tune the hybrid model

precision_scores = []
recall_scores = []
ndcg_scores = []

for _ in range(num_tests):
    avg_precision, avg_recall, avg_ndcg = evaluate_recommender(user_item_matrix, users, items, topn, popularity_weight)
    precision_scores.append(avg_precision)
    recall_scores.append(avg_recall)
    ndcg_scores.append(avg_ndcg)

# Print the evaluation results
print(f"Evaluation Results (Top-{topn} Recommendations):")
print(f"Precision: {np.mean(precision_scores):.4f} +/- {np.std(precision_scores):.4f}")
print(f"Recall: {np.mean(recall_scores):.4f} +/- {np.std(recall_scores):.4f}")
print(f"NDCG: {np.mean(ndcg_scores):.4f} +/- {np.std(ndcg_scores):.4f}")