# Fashion Recommendation and Virtual Try-On System

Welcome to the **Fashion Recommendation and Virtual Try-On System**! This repository houses a state-of-the-art application that combines advanced machine learning techniques and an intuitive user interface to recommend fashion items and enable virtual try-ons. Whether you're a fashion enthusiast or a data science buff, this project showcases the intersection of AI, computer vision, and style.  

---

## Key Features

### 1. **Recommendation System**
- **ResNet50** is used to extract meaningful features from images, ensuring accurate and efficient similarity detection.
- Nearest Neighbors algorithm suggests the top 5 most visually similar items.
- Explore various embeddings and feature visualizations like **TSNE** for deeper insights.

### 2. **Virtual Try-On**
- Integrates with the **KlingAI API** to enable virtual try-on functionality.
- Upload your image and a garment image, and let the system combine them seamlessly.
- Process includes real-time task management, image retrieval, and enhancement.

### 3. **Data Exploration and Visualization**
- **Confusion Matrix**, **ROC Curve**, and **Precision-Recall Curve** provide performance metrics for the recommendation system.
- **TensorBoard Dashboards** track training and validation metrics, aiding hyperparameter tuning.
- **SHAP Values** highlight key features driving recommendations, such as color and style.

---

## Architecture Overview

### Technologies Used:
- **Frontend**: Built with [Streamlit](https://streamlit.io/) for an interactive and user-friendly experience.
- **Backend**: Utilizes **TensorFlow**, **Keras**, and **Sklearn** for feature extraction, recommendation, and performance evaluation.
- **API Integration**: Powered by **KlingAI API** for advanced virtual try-on capabilities.
- **Visualization Tools**: TensorBoard and Matplotlib for insights into model performance and data trends.

### Workflow:
1. **Image Processing**: Leverages ResNet50 to extract robust features from uploaded images.
2. **Recommendation Generation**: Uses Nearest Neighbors to find visually similar fashion items.
3. **Virtual Try-On**: Encodes images, communicates with KlingAI API, and retrieves augmented outputs.

---

## Getting Started

### Prerequisites

1. Install the following Python libraries:
   ```bash
   pip install streamlit tensorflow scikit-learn matplotlib pillow requests pyjwt
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/syedanida/FashionRecommender-and-VirtualTryOn.git
   ```
3. Ensure the following files are in the repository:
   - `image_features_embedding.pkl`: Pre-computed feature embeddings.
   - `img_files.pkl`: Metadata of fashion images.

### Running the Application

1. Navigate to the project directory:
   ```bash
   cd FashionRecommender-and-VirtualTryOn
   ```
2. Run app.py:
   ```bash
   python run.py
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the application in your browser at `http://localhost:8501`.

---

## Screenshots

### **Recommendation System**
![Recommendation System UI](https://github.com/syedanida/FashionRecommender-and-VirtualTryOn/blob/main/demoimage1.png)

![image](https://github.com/syedanida/FashionRecommender-and-VirtualTryOn/blob/main/demoimage2.png)

### **Virtual Try-On**
![Virtual Try-On UI](https://github.com/syedanida/FashionRecommender-and-VirtualTryOn/blob/main/demoimage3.png)
![Virtual Try-On UI](https://github.com/syedanida/FashionRecommender-and-VirtualTryOn/blob/main/demoimage4.png)

---

## How It Works

### **Recommendation System**
- Upload a fashion item image.
- The model extracts features using ResNet50 and compares them against the pre-computed embeddings.
- The top 5 visually similar items are displayed with images and links.

### **Virtual Try-On**
- Upload your photo and a garment image.
- The system sends the images to the KlingAI API for processing.
- The combined output is returned and displayed in real-time.

---

## Visualizations

1. **Confusion Matrix**
   - Highlights true positives and negatives, showcasing model accuracy.
2. **ROC Curve**
   - AUC = 0.95, demonstrating excellent model performance.
3. **TSNE Plot**
   - Visualizes ResNet50 embeddings for meaningful clustering of apparel items.
4. **TensorBoard Metrics**
   - Tracks training loss and validation accuracy to guide optimization.

---

## Insights and Learnings

- **Model Performance**: Achieved high accuracy with minimal misclassification.
- **Key Features**: SHAP analysis confirms that attributes like color, material, and style drive effective recommendations.
- **Scalability**: Modular architecture supports adding new models or APIs seamlessly.

---

## Future Enhancements

- **Seasonal Recommendations**: Suggest outfits based on seasons and current trends.
- **Outfit Planning**: Integrate tools for assembling complete looks.
- **Multimodal Enhancements**: Incorporate textual descriptions and user preferences for personalized results.

---

## Acknowledgments

- [KlingAI](https://klingai.com) for the API integration.
- [Streamlit](https://streamlit.io/) for the intuitive frontend framework.
- TensorFlow and Keras for powering the deep learning models.

---

Enjoy exploring and contributing to the **Fashion Recommendation and Virtual Try-On System**!
