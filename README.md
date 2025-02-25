# 📦 Brand-UGBP-Incongruency
A machine learning pipeline to measure the **incongruency** between **user-generated branded photos (UGBP) and brand values** using **AI-driven image processing and text embedding techniques**.

---

## 🚀 **Project Overview**
This repository provides a framework to:
- **Extract** and **process** images from social media posts.
- **Generate** image descriptions using OpenAI's GPT-4 Vision API.
- **Compute** brand values from text-based brand associations.
- **Compare** brand and image descriptions using **embedding models**.
- **Measure** incongruency using **Euclidean distance** between vectors.

---

## 📁 **Project Structure**
📦 brand-ugbp-incongruency ┣ 📂 data ┃ ┣ cleaned_df.json # Main dataset containing brand details ┃ ┣ image_descriptions.json # Processed image descriptions ┃ ┣ brand_embeddings.json # Processed brand embeddings ┃ ┣ post_incongruency.json # Final dataset with Euclidean distances ┃ ┗ image/ # Folder containing images ┣ 📂 scripts ┃ ┣ image_processing.py # Image processing and description generation ┃ ┣ text_embedding.py # Brand text embedding and processing ┃ ┗ measure_incongruency.py # Compute Euclidean distances between images & brands ┣ 📜 requirements.txt # Required dependencies ┣ 📜 .gitignore # Ignore sensitive data & API keys ┗ 📜 README.md # Documentation  

📊 Methodology
Image Processing: Extracts key attributes like colorfulness, brightness, logo presence, and human faces.
Brand Association Extraction: Uses LLMs (GPT-4) to generate brand values.
Embedding Computation: Uses OpenAI's text-embedding model to convert text into numerical representations.
Incongruency Measurement: Computes Euclidean distance between embeddings to measure visual-brand incongruency

👩‍💻 Author
Developed by Majedeh Esmizadeh