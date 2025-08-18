import os
import pandas as pd

# Construct absolute path to user interaction CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Backend/services
DATA_FILE = os.path.join(BASE_DIR, "..", "..", "Data", "merged_user_interaction_data.csv")
DATA_FILE = os.path.normpath(DATA_FILE)

print(f"Looking for user interactions file at: {DATA_FILE}")
print("Exists:", os.path.exists(DATA_FILE))

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"User interactions file not found at: {DATA_FILE}")

# Load CSV once
user_interactions_df = pd.read_csv(DATA_FILE)

# Convert numeric user_ids to 'user_XXX' string format if needed
if user_interactions_df['user_id'].dtype != 'object':
    user_interactions_df['user_id'] = user_interactions_df['user_id'].apply(lambda x: f"user_{int(x):03d}")

def validate_user(user_id: str) -> bool:
    if not user_id.startswith("user_"):
        return False
    suffix = user_id[5:]
    if len(suffix) != 3 or not suffix.isdigit():
        return False
    return user_id in user_interactions_df["user_id"].unique()

def get_user_history(user_id: str) -> pd.DataFrame:
    return user_interactions_df[user_interactions_df["user_id"] == user_id]
