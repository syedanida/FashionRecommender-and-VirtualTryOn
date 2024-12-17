import streamlit as st
import tensorflow as tf
import os
import cv2
import numpy as np
import random
import base64
import requests
import json
import time
import jwt
import logging
import pickle
from numpy.linalg import norm
from tqdm import tqdm
from PIL import Image
from typing import Optional, Dict, Any, Union, Tuple
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors



def unzip_file(zip_path: str, extract_path: str) -> None:
    tqdm.write(f"Unzipping file {str}...")
    
    """
    Unzip a file to the specified path
    Args:
        zip_path: Path to the zip file
        extract_path: Path where to extract the contents
    """
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# if fashion_small folder does not exists in dir unzip  ./fashion_small.zip
if not os.path.exists('fashion_small'):
    unzip_file('fashion_small.zip', '.')


if not os.path.exists('styles'):
    unzip_file('styles.zip', '.')




# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])





def create_pkl():
    tqdm.write("Creating Embeddings and Product details...")
    img_files = []
    for fashion_image in os.listdir('fashion_small/images'):
        images_path = os.path.join('fashion_small/images', fashion_image)
        
        # Load and parse the corresponding style JSON
        json_path = os.path.join('styles', fashion_image.replace('.jpg', '.json'))
        with open(json_path, 'r') as f:
            style_data = json.load(f)['data']
            
        # Extract relevant information
        product_info = {
            "image_path": images_path,
            "product_url": style_data.get('landingPageUrl', ''),
            "brand": style_data.get('brandName', ''),
            "product_name": style_data.get('productDisplayName', ''),
            "category": style_data.get('displayCategories', ''),
            "color": style_data.get('baseColour', ''),
            "season": style_data.get('season', '')
        }
        img_files.append(product_info)

    # Extract features for all images
    image_features = []
    batch_size = 64
    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i:i + batch_size]
        batch_images = []
        for files in batch_files:
            img = image.load_img(files["image_path"], target_size=(224, 224))
            img_array = image.img_to_array(img)
            batch_images.append(img_array)
        
        batch_images = np.array(batch_images)
        preprocessed_batch = preprocess_input(batch_images)
        batch_features = model.predict(preprocessed_batch)
        
        # Normalize each feature vector in the batch
        for features in batch_features:
            features_flat = features.flatten()
            features_normalized = features_flat / norm(features_flat)
            image_features.append(features_normalized)
        
        if i % (batch_size * 10) == 0:
            tqdm.write(f"Processed {i}/{len(img_files)} images")
    # Save the features and image metadata
    pickle.dump(image_features, open("image_features_embedding.pkl", "wb"))
    pickle.dump(img_files, open("img_files.pkl", "wb"))
    

# if image_features_embedding and img_files are not present in directory, create them
if not os.path.exists("image_features_embedding.pkl") or not os.path.exists("img_files.pkl"):
    create_pkl()
    


# Load pre-computed features
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))



class KlingAIClient:
    def __init__(self, access_key: str, secret_key: str, base_url: str):
        self.access_key = "8848da2d9445405283f262c16b3173e7"
        self.secret_key = "22bd1c413003442786e970df07525db4"
        self.base_url = "https://api.klingai.com"
        self.logger = logging.getLogger(__name__)

    def _generate_jwt_token(self) -> str:
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        return jwt.encode(payload, self.secret_key, headers=headers)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self._generate_jwt_token()}"
        }

    def try_on(self, person_img: np.ndarray, garment_img: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
        tqdm.write("Trying on ..")
        if person_img is None or garment_img is None:
            raise ValueError("Empty image")
            
        encoded_person = cv2.imencode('.jpg', cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))[1]
        encoded_person = base64.b64encode(encoded_person.tobytes()).decode('utf-8')
        
        encoded_garment = cv2.imencode('.jpg', cv2.cvtColor(garment_img, cv2.COLOR_RGB2BGR))[1]
        encoded_garment = base64.b64encode(encoded_garment.tobytes()).decode('utf-8')

        url = f"{self.base_url}/v1/images/kolors-virtual-try-on"
        data = {
            "model_name": "kolors-virtual-try-on-v1",
            "cloth_image": encoded_garment,
            "human_image": encoded_person,
            "seed": seed
        }

        try:
            response = requests.post(
                url, 
                headers=self._get_headers(),
                json=data,
                timeout=50
            )
            response.raise_for_status()
            
            result = response.json()
            task_id = result['data']['task_id']
            
            time.sleep(4)
            
            for attempt in range(12):
                try:
                    url = f"{self.base_url}/v1/images/kolors-virtual-try-on/{task_id}"
                    response = requests.get(url, headers=self._get_headers(), timeout=20)
                    response.raise_for_status()
                    
                    result = response.json()
                    status = result['data']['task_status']
                    
                    if status == "succeed":
                        output_url = result['data']['task_result']['images'][0]['url']
                        img_response = requests.get(output_url)
                        img_response.raise_for_status()
                        
                        nparr = np.frombuffer(img_response.content, np.uint8)
                        result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        return result_img, "Success"
                    elif status == "failed":
                        return None, f"Error: {result['data']['task_status_msg']}"
                        
                except requests.exceptions.ReadTimeout:
                    if attempt == 11:
                        return None, "Request timed out"
                        
                time.sleep(1)
                
            return None, "Processing took too long"
            
        except Exception as e:
            self.logger.error(f"Error in try_on: {str(e)}")
            return None, f"Error: {str(e)}"


# Function to extract features from image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # Normalizing
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized


def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(features_list)
    distance, indices = neighbors.kneighbors([features])
    return indices


def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploader"):
            os.makedirs("uploader")
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


def main():
    if 'selected_garment' not in st.session_state:
        st.session_state['selected_garment'] = None

    st.title('Fashion Recommendation System with Virtual Try-On')
    
    # Recommendation Section
    st.header("Get Similar Fashion Recommendations")
    uploaded_file = st.file_uploader("Choose your fashion image", key="fashion_image")
    
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            show_image = Image.open(uploaded_file)
            resized_image = show_image.resize((400, 400))
            st.image(resized_image)
            
            features = extract_features(os.path.join("uploader", uploaded_file.name), model)
            indices = recommend(features, features_list)
            
            st.subheader("Similar Fashion Items:")
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.write(f"#{idx+1}")
                    img_idx = indices[0][idx]
                    product_info = img_files_list[img_idx]

                    # Display image
                    st.image(product_info["image_path"])
                    
                    # Display product details
                    # st.write(f"**{product_info['brand']}**")
                    st.write(product_info['product_name'])
                    
                    # Additional details in expandable section
                    with st.expander("More Details"):
                        st.write(f"Category: {product_info['category']}")
                        st.write(f"Color: {product_info['color']}")
                        st.write(f"Season: {product_info['season']}")
                    
                    # Product and Try-On buttons
                    # col1, col2 = st.columns(2)
                    st.write(f"[View Product](https://www.myntra.com/{product_info['product_url']})")
                    if st.button(f"Try On #{idx+1}"):
                        st.session_state['selected_garment'] = product_info["image_path"]

    
    if st.session_state['selected_garment']:
        # Virtual Try-On Section
        st.markdown("---")
        st.header("Virtual Try-On")
        
        client = KlingAIClient(
            access_key="YOUR_ACCESS_KEY",
            secret_key="YOUR_SECRET_KEY",
            base_url="https://api.klingai.com"
        )
        st.write("Selected Garment for Try-On:")
        st.image(st.session_state['selected_garment'])
        person_image = st.file_uploader("Upload Person Image", type=['jpg', 'jpeg', 'png'])

        if person_image:
            seed = random.randint(0, 999999)
            if st.button("Generate Virtual Try-On"):
                person_img = np.array(Image.open(person_image))
                garment_img = np.array(Image.open(st.session_state['selected_garment']))
                
                result_img, status = client.try_on(person_img, garment_img, seed)
                if result_img is not None:
                    st.image(result_img, caption="Virtual Try-On Result")
                else:
                    st.error(status)


if __name__ == "__main__":
    main()

