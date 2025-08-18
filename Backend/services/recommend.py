import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


import os
import pandas as pd

import os
import pandas as pd

current_file = os.path.abspath(__file__)               # Backend/services/recommend.py
project_root = os.path.dirname(os.path.dirname(current_file))    # Go up two levels

# Path to Data folder one level above Backend (project root)
data_dir = os.path.join(os.path.dirname(project_root), "Data")

merged_csv = os.path.join(data_dir, "merged_user_interaction_data.csv")
cleaned_csv = os.path.join(data_dir, "cleaned_data.csv")

merged_df = pd.read_csv(merged_csv)
cleaned_data = pd.read_csv(cleaned_csv)



def userwise_split(df, test_frac=0.2, seed=42):
    train_rows, test_rows = [], []
    grouped = df.groupby('user_id')
    for user_id, group in grouped:
        if len(group) < 2:
            train_rows.append(group)
            continue
        train, test = train_test_split(group, test_size=test_frac, random_state=seed)
        train_rows.append(train)
        test_rows.append(test)
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df

def compute_similarity_matrices(train_df):
    user_item_matrix = train_df.pivot_table(index='user_id', columns='product_id', values='score', fill_value=0)
    user_similarity = pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    item_similarity = pd.DataFrame(
        cosine_similarity(user_item_matrix.T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    return user_item_matrix, user_similarity, item_similarity

def recommend_items_hybrid(user_id, train_matrix, user_sim, item_sim, top_k=5, alpha=0.5):
    if user_id not in train_matrix.index:
        return []
    user_vec = train_matrix.loc[user_id]
    interacted = user_vec[user_vec > 0].index.tolist()
    if not interacted:
        return []

    # User-based prediction
    sim_scores = user_sim.loc[user_id].drop(user_id, errors='ignore')
    weighted_user_sum = (sim_scores.values.reshape(-1,1) * train_matrix.loc[sim_scores.index]).sum(axis=0)
    weighted_user_sum = pd.Series(weighted_user_sum, index=train_matrix.columns)

    # Item-based prediction
    weighted_item_sum = pd.Series(0, index=train_matrix.columns, dtype=float)
    for item in interacted:
        weighted_item_sum += item_sim.loc[item]
    weighted_item_sum = weighted_item_sum.drop(interacted, errors='ignore')

    # Combine user and item predictions
    combined_scores = (alpha * weighted_user_sum) + ((1 - alpha) * weighted_item_sum)
    recommended = combined_scores.sort_values(ascending=False).head(top_k).index.tolist()
    return recommended

def precision_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)

def get_product_details_for_recommendations(recommended_product_ids, product_catalog):
    recommended_details = product_catalog[product_catalog['product_id'].isin(recommended_product_ids)].copy()
    recommended_details = recommended_details.set_index('product_id').loc[recommended_product_ids].reset_index()
    return recommended_details

# Prepare training and testing data and similarity matrices once (can call this in API startup)
train_df, test_df = userwise_split(merged_df)
user_item_matrix, user_similarity, item_similarity = compute_similarity_matrices(train_df)
test_users = test_df['user_id'].unique()
test_truth = test_df.groupby('user_id')['product_id'].apply(set).to_dict()

def evaluate_hybrid_cf(alpha=0.5, top_k=5):
    precisions, recalls = [], []
    for user in test_users:
        recs = recommend_items_hybrid(user, user_item_matrix, user_similarity, item_similarity, top_k=top_k, alpha=alpha)
        truth = test_truth.get(user, set())
        precisions.append(precision_at_k(recs, truth, top_k))
        recalls.append(recall_at_k(recs, truth, top_k))
    return np.mean(precisions), np.mean(recalls)

# Example evaluation:
if __name__ == "__main__":
    precision, recall = evaluate_hybrid_cf(alpha=0.5, top_k=5)
    print(f"Hybrid CF (alpha=0.5): Precision@5 = {precision:.4f}")
    print(f"Hybrid CF (alpha=0.5): Recall@5 = {recall:.4f}")

    sample_user = test_users[110]
    recommended_ids = recommend_items_hybrid(sample_user, user_item_matrix, user_similarity, item_similarity, top_k=5, alpha=0.5)
    detailed_recs = get_product_details_for_recommendations(recommended_ids, cleaned_data)
    print(f"Recommendations for user {sample_user}:")
    print(detailed_recs[['product_name', 'product_type', 'brand', 'price']])

