import gzip
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Preprocess the text data
def preprocess_text(text):
    # Tokenize, lowercase, and remove stopwords
    # You can use libraries like NLTK or spaCy for more advanced preprocessing
    return text.lower().split()

# Create feature representations using TF-IDF
def create_feature_matrix(meta_data):
    corpus = [' '.join(preprocess_text(item['title'])) for item in meta_data]
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(corpus)
    return feature_matrix

# Calculate cosine similarity between products
def calculate_similarity(feature_matrix):
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix

# Get top-N recommendations for a given product
def get_recommendations(parent_asin, similarity_matrix, meta_data, top_n=5):
    idx = next((i for i, item in enumerate(meta_data) if item.get('parent_asin') == parent_asin), None)
    if idx is None:
        return []
    similar_indices = similarity_matrix[idx].argsort()[::-1][1:top_n+1]
    similar_products = [(meta_data[i]['parent_asin'], meta_data[i]['title']) for i in similar_indices]
    return similar_products

# Example usage
meta_file_path = 'meta_All_Beauty.jsonl.gz'
review_file_path = 'All_Beauty.jsonl.gz'

meta_data = load_meta_data(meta_file_path)
review_data = load_review_data(review_file_path)

# Extract the first 100 unique parent_asin values from meta_data
parent_asins = list(set(item.get('parent_asin') for item in meta_data if item.get('parent_asin')))[:100]

# Filter meta_data based on the selected parent_asin values
filtered_meta_data = [item for item in meta_data if item.get('parent_asin') in parent_asins]

feature_matrix = create_feature_matrix(filtered_meta_data)
similarity_matrix = calculate_similarity(feature_matrix)

# Print the first 100 unique parent_asin values
print("First 100 unique parent_asin values:")
for parent_asin in parent_asins:
    print(parent_asin)

# Get recommendations for each parent_asin
for parent_asin in parent_asins:
    recommendations = get_recommendations(parent_asin, similarity_matrix, filtered_meta_data)
    if recommendations:
        print(f"\nRecommendations for parent_asin {parent_asin}:")
        for rec_parent_asin, rec_title in recommendations:
            print(f"- {rec_parent_asin}: {rec_title}")
    else:
        print(f"\nNo recommendations found for parent_asin {parent_asin}")