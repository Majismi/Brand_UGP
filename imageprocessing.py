###### in this code we process image characteristic like: colorfulness, brightness, human_presence and logo_size
#### use google cloud vision for logo detection and face detection
#### write function and use python packges for colorfulness and brightness

import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from google.cloud import vision

# Set Google Cloud Vision API Credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'path_credentials_google_cloud'

client = vision.ImageAnnotatorClient()

# Define the folder containing images
folder_path = r"path_images"


####colorfulness function, RGB
def calculate_colorfulness(image_opened):
    
    img = np.array(image_opened)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)

    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)

    # Compute final colorfulness score
    std_root = np.sqrt(rg_std ** 2 + yb_std ** 2)
    mean_root = np.sqrt(rg_mean ** 2 + yb_mean ** 2)
    colorfulness = std_root + (0.3 * mean_root)
    
    return colorfulness


#Compute brightness metric of an image by converting it to grayscale.
def calculate_brightness(image_opened):
    
    img = image_opened.convert('L')  
    img_data = np.asarray(img)
    brightness = np.mean(img_data)
    return brightness


###detect faces, logos and labels from an image using google cloud vision
def detect_logos_faces_labels(image_path):
   
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
        image = vision.Image(content=content)

    # Logo detection
    logo_response = client.logo_detection(image=image)
    logos = logo_response.logo_annotations
    logo_data = [(logo.description, logo.score) for logo in logos] if logos else None

    # Face detection
    face_response = client.face_detection(image=image)
    num_faces = len(face_response.face_annotations)

    # Label detection
    label_response = client.label_detection(image=image)
    labels = [(label.description, label.score) for label in sorted(label_response.label_annotations, key=lambda l: l.score, reverse=True)[:1]]
    
    return logo_data, num_faces, labels

#Process all images

final_df = []
count = 0

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    
    try:
        
        with Image.open(image_path) as image_file:
            image_width, image_height = image_file.size
            image_area = image_width * image_height
            
            
            colorfulness_score = calculate_colorfulness(image_file)
            brightness_score = calculate_brightness(image_file)
        
        
        logo_info, face_count, labels_info = detect_logos_faces_labels(image_path)
        label_description, label_score = labels_info[0] if labels_info else (None, None)
        brand_name, logo_score = logo_info[0] if logo_info else (None, None)

        ### Append results to dataframe
        final_df.append((filename, image_area, brand_name, logo_score, face_count, colorfulness_score, brightness_score, label_description, label_score))

        count += 1
        print(f"Processed {count} images")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")


df = pd.DataFrame(final_df, columns=['Filename', 'Image_Area', 'Brand_Name', 'Logo_Score', 'Face_Count', 'Colorfulness', 'Brightness', 'Label', 'Label_Score'])


output_path = r'file_path.json'
df.to_json(output_path, orient='records')

print(f"Processed data saved to: {output_path}")