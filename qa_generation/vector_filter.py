import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the OpenAI Embeddings model
embedder = OpenAIEmbeddings(model="text-embedding-ada-002",
                            openai_api_key="")
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Function to precompute embeddings for all rows in a DataFrame column
def precompute_embeddings(df, text_column):
    embeddings = df[text_column].apply(embedder.embed_query)
    return embeddings

# Function to filter DataFrame based on cosine similarity threshold

def create_similarity_matrix(embeddings):
    # Convert embeddings to a NumPy array for efficient computation
    embeddings_array = np.array(embeddings.tolist())
    similarity_matrix = cosine_similarity(embeddings_array)
    return similarity_matrix

def filter_dataframe(df, threshold):
    # Add a 'keep_row' column to the dataframe, initialized to True
    df = df.copy()
    df['keep_row'] = True

    for (idx1, row1), (idx2, row2) in combinations(df.iterrows(), 2):
        # Skip if either row is already marked to be dropped
        if not df.at[idx1, 'keep_row'] or not df.at[idx2, 'keep_row']:
            continue

        # Reshape embeddings to 2D arrays
        #emb1 = np.array(row1['Embedding']).reshape(1, -1)
        #emb2 = np.array(row2['Embedding']).reshape(1, -1)

        #sim = cosine_similarity(emb1, emb2)[0][0]
        result = cosine_similarity([row1['Embedding'], row2['Embedding']])
        sim = result[0, 1]  # or result[1, 0] as both are the same

        if sim > threshold:
            # Choose one of the rows to drop
            df.at[idx2, 'keep_row'] = False  # or idx1, based on your criteria

    # Filter the dataframe using the 'keep_row' column
    
    true_count = df['keep_row'].value_counts().get(True, 0)
    false_count = df['keep_row'].value_counts().get(False, 0)

    filtered_df = df[df['keep_row']].drop(columns='keep_row')
    return filtered_df
