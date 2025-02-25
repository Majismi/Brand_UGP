import os
import json
import time
import base64
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration
IMAGE_FOLDER = "data/image"
OUTPUT_FILE = "data/image_descriptions.json"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def encode_image(image_path):
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_image_description(image_path):
    
    base64_image = encode_image(image_path)
    prompt = "Describe the image using 5 adjectives."

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "max_tokens": 100,
        "seed": 12345
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload)
    response_data = response.json()

    return response_data.get("choices", [{}])[0].get("message", {}).get("content", "error")

def process_images():

    results = []
    
    for count, filename in enumerate(os.listdir(IMAGE_FOLDER), start=1):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        description = generate_image_description(image_path)
        results.append({"Filename": filename, "Image_Description": description})
        
        print(f"Processed {count}: {filename}")
        time.sleep(60)  

        
        if count % 10 == 0:
            save_results(results)

    save_results(results)
    print(f"Processing complete. Data saved to {OUTPUT_FILE}")

def save_results(results):
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    process_images()