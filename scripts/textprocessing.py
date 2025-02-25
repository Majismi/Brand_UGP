import pandas as pd
import re
import emoji
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat

# Load the dataset
data_path = r"your_data.json"
data = pd.read_json(data_path, encoding='utf-8')

# Define brand names for mention and hashtag tracking
brands = ['cocacola', 'pepsi', 'nestle', 'redbull', 'drpepper']

################ MENTIONS ####################

# Count mentions
data['mention_count'] = data['mentions'].apply(lambda x: len([item for item in x.strip().split(',') if item.strip()]))

# create dummy variables for tracking mentioned brands
def mark_mentions(mention_text):
    mentioned_brands = []
    mention_text = mention_text.lower()  # Convert text to lowercase to match brands

    for brand in brands:
        if re.search(rf'\b{brand}\b', mention_text):
            mentioned_brands.append(brand)

    return ', '.join(mentioned_brands) if mentioned_brands else None

data['brands_mentioned'] = data['mentions'].apply(mark_mentions)

for brand in brands:
    data[f'{brand}_mentioned'] = data['brands_mentioned'].apply(lambda x: 1 if brand in str(x) else 0)


################ TAGS ####################

# create dummy variable for tagged posts
data['is_tagged'] = data['taggedUsers'].apply(lambda x: 1 if x.strip() else 0)


################ HASHTAGS ####################

# Count number of hashtags
data['hashtag_count'] = data['hashtags'].apply(lambda x: len([item for item in x.strip().split(',') if item.strip()]))



################ CAPTIONS ####################

# Count number of words in the caption
data['caption_word_count'] = data['caption'].apply(lambda x: len(x.split()))

# Count number of emojis in the caption
data['emoji_count'] = data['caption'].apply(lambda x: emoji.emoji_count(x))

# Sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()
data['vader_sentiment'] = data['caption'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Text complexity using Gunning Fog Index
data['caption_complexity'] = data['caption'].apply(lambda x: textstat.gunning_fog(x))


# ############### SAVE PROCESSED DATA ####################

output_folder = r'folder-path'
output_file = 'cleaned_text.json'
output_path = os.path.join(output_folder, output_file)

data.to_json(output_path, orient='records', lines=True)

print(f"Processed data saved to: {output_path}")