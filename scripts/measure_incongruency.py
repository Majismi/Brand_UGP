import os
import json
import numpy as np
import pandas as pd

# File paths
image_embed_file = "data/image_embeddings.json"
brand_embed_file = "data/brand_embeddings.json"
Output_file = "data/post_incongruency.json"

# Load data
post_embed = pd.read_json(image_embed_file, lines=True)
brand_embed = pd.read_json(brand_embed_file, lines=True)

post_embed["euclidean_distance"] = 0.0

for index, post_row in post_embed.iterrows():
    brand_row = brand_embed[brand_embed["Brand"] == post_row["Brand"]].iloc[0]
    
    brand_embedding = np.array(brand_row["embedding_brand"])
    post_embedding = np.array(post_row["embedding_image"])

    distance = np.linalg.norm(brand_embedding - post_embedding)
    post_embed.at[index, "euclidean_distance"] = distance

post_embed.to_json(Output_file, orient="records", lines=True)
print(f"Incongruency scores saved to {Output_file}")