import os
import json
import openai
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
Brand_Data = "data/cleaned_df.json"
Image_Data = "data/image_description.json"
Output_File = "data/brand_embeddings.json"
Output_File2 = "data/image_embeddings.json"

def preprocess_text(text):
    
    text = re.sub(r"\d+", "", text.replace("\n", " ").replace(".", ""))
    return re.sub(r"\s+", " ", text).strip()

def get_embedding(text):

    response = openai.Embedding.create(model="text-embedding-3-large", input=text)
    return response["data"][0]["embedding"]

def process_brand_embeddings():
    
    df = pd.read_json(Brand_Data)
    df["processed_text"] = df["Brand_value"].apply(preprocess_text)
    df["embedding_brand"] = df["processed_text"].apply(get_embedding)
    
    df.to_json(Output_File, orient="records", lines=True)
    print(f"Brand embeddings saved to {Output_File}")

def process_image_embeddings():
    
    df2 = pd.read_json(Image_Data)
    df2["processed_text"] = df2["Image_Description"].apply(preprocess_text)
    df2["embedding_image"] = df2["processed_text"].apply(get_embedding)
    
    df2.to_json(Output_File, orient="records", lines=True)
    print(f"Brand embeddings saved to {Output_File2}")

if __name__ == "__main__":
    process_brand_embeddings()
    process_image_embeddings()
