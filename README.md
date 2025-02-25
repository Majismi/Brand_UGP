# 📦 Brand-UGBP-Incongruency
A machine learning pipeline to measure the **incongruency** between **user-generated branded photos (UGBP) and brand values** using **AI-driven image processing and text embedding techniques**.

---

**Project Overview**
This repository provides a framework to:
- **Extract** and **process** images from social media posts.
- **Generate** image descriptions using OpenAI's GPT-4 Vision API.
- **Compute** brand values from text-based brand associations.
- **Compare** brand and image descriptions using **embedding models**.
- **Measure** incongruency using **Euclidean distance** between vectors.

---
  

Methodology
Image Processing: Extracts key attributes like colorfulness, brightness, logo presence, and human faces.
Brand Association Extraction: Uses LLMs (GPT-4) to generate brand values.
Embedding Computation: Uses OpenAI's text-embedding model to convert text into numerical representations.
Incongruency Measurement: Computes Euclidean distance between embeddings to measure visual-brand incongruency

Author
Developed by Majedeh Esmizadeh