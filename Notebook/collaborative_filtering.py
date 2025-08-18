

import pandas as pd
df=pd.read_csv(r"../Data/cleaned_data.csv")
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import random




merged_df=pd.read_csv("../Data/merged_user_interaction_data.csv")




# --- Your existing userwise_split function ---
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

# --- Hybrid recommendation function ---
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

    # Combine
    combined_scores = (alpha * weighted_user_sum) + ((1 - alpha) * weighted_item_sum)
    recommended = combined_scores.sort_values(ascending=False).head(top_k).index.tolist()
    return recommended

# --- Precision and Recall functions ---
def precision_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)

# ----------------------------------
# Suppose you have `merged_df` with columns ["user_id", "product_id", "score", ...]
# Example usage:

# 1. Split data
train_df, test_df = userwise_split(merged_df)

# 2. Create user-item rating matrix from train data
user_item_matrix = train_df.pivot_table(index='user_id', columns='product_id', values='score', fill_value=0)

# 3. Compute similarity matrices from train data
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

# 4. Prepare test users and ground truth dict
users = test_df['user_id'].unique()
test_truth = test_df.groupby('user_id')['product_id'].apply(set).to_dict()

def get_product_details_for_recommendations(recommended_product_ids, cleaned_data):
    recommended_details = cleaned_data[cleaned_data['product_id'].isin(recommended_product_ids)].copy()
    recommended_details = recommended_details.set_index('product_id').loc[recommended_product_ids].reset_index()
    return recommended_details

# 5. Evaluate hybrid CF
alpha = 0.5  # user-user and item-item blending weight
precisions, recalls = [], []

for user in users:
    recs = recommend_items_hybrid(user, user_item_matrix, user_similarity, item_similarity, top_k=5, alpha=alpha)
    detailed_recs = get_product_details_for_recommendations(recs, df)
    print(f"User: {user} Recommendations:\n", detailed_recs[['product_id', 'product_name', 'brand', 'price', 'product_type']])
    truth = test_truth.get(user, set())
    precisions.append(precision_at_k(recs, truth, 5))
    recalls.append(recall_at_k(recs, truth, 5))

print(f"Hybrid CF (alpha={alpha}): Precision@5 = {np.mean(precisions):.4f}")
print(f"Hybrid CF (alpha={alpha}): Recall@5 = {np.mean(recalls):.4f}")




sample_user = users[220]
recommended_ids = recommend_items_hybrid(sample_user, user_item_matrix, user_similarity, item_similarity, top_k=5, alpha=0.5)
detailed_recs = get_product_details_for_recommendations(recommended_ids, df)
print(detailed_recs[['product_name', 'product_type', 'brand', 'price']])













