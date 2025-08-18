import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===== Construct absolute paths relative to this script =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Go two levels up to reach the root folder, then Embeddings
EMB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Embeddings"))

PKL_PATH = os.path.join(EMB_DIR, "product_data_with_embeddings.pkl")
NPY_PATH = os.path.join(EMB_DIR, "product_embeddings.npy")

print(f"Looking for pickle file at: {PKL_PATH}")
print(f"Exists: {os.path.exists(PKL_PATH)}")
print(f"Looking for embeddings file at: {NPY_PATH}")
print(f"Exists: {os.path.exists(NPY_PATH)}")

_df = None
_embeddings = None
_model = None


def _ensure_loaded():
    global _df, _embeddings, _model
    if _df is None or _embeddings is None:
        if not (os.path.exists(PKL_PATH) and os.path.exists(NPY_PATH)):
            raise FileNotFoundError(
                f"Required embedding files not found at:\n{PKL_PATH}\n{NPY_PATH}\n"
                "Please ensure the files exist or run the embedding generation script."
            )
        _df = pd.read_pickle(PKL_PATH)
        _embeddings = np.load(NPY_PATH)
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")


def search_products(query: str, filters=None, top_k=10):
    _ensure_loaded()
    query_vec = _model.encode([query])
    sims = cosine_similarity(query_vec, _embeddings)[0]
    ranked = sims.argsort()[::-1]
    results = _df.iloc[ranked].copy()
    results["similarity"] = sims[ranked]

    if filters:
        for key, values in filters.items():
            if key in results.columns and values:
                results = results[results[key].isin(values)]

    cols = [c for c in ["product_name", "brand", "product_type", "description", "price", "similarity"] if c in results.columns]
    return results.loc[:, cols].head(top_k)
if __name__ == "__main__":
    results = search_products("face wash", top_k=5)
    print(results)
