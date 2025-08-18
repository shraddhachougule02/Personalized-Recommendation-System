
import pandas as pd
df=pd.read_csv(r"../Data/cleaned_data.csv")
df.head()


def combine_text(row):
    return (
        f"Product: {row['product_name']}. "
        f"Type: {row['product_type']}. "
        f"Brand: {row['brand']}. "
        f"Description: {row['description']}. "
        f"Skin Types: {', '.join(row['skintype_list'])}. "
        f"Effects: {', '.join(row['notable_effects_list'])}."
    )

df['combined_text'] = df.apply(combine_text, axis=1)

df['combined_text']


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')



#!pip install sentence-transformers

# convert the column to a list of strings
corpus = df['combined_text'].tolist() 

embeddings = model.encode(corpus, show_progress_bar=True)

df['embedding'] = embeddings.tolist()



import numpy as np
np.save('../Embeddings/product_embeddings.npy', embeddings)
df.to_pickle('../Embeddings/product_data_with_embeddings.pkl')


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_search(query, model, df, embeddings, top_k=5):
    # Encode the user query into a vector
    query_vec = model.encode([query])

    # Calculate cosine similarity between the query vector and product embeddings
    similarities = cosine_similarity(query_vec, embeddings)[0]

    # Get the indices of top-k most similar products
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Return the top-k products with similarity scores
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results[['product_name', 'brand', 'product_type', 'description', 'similarity']]


query="hydrating moisturizer for dry skin"
output=semantic_search(query,model,df,embeddings,top_k=5)
print(output)








