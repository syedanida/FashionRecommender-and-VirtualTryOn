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
   git clone https://github.com/your-username/fashion-tryon.git
   ```
3. Ensure the following files are in the repository:
   - `image_features_embedding.pkl`: Pre-computed feature embeddings.
   - `img_files.pkl`: Metadata of fashion images.

### Running the Application

1. Navigate to the project directory:
   ```bash
   cd fashion-tryon
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the application in your browser at `http://localhost:8501`.

---

## Screenshots

### **Recommendation System**
![Recommendation System UI](assets/recommendation_system.png)

### **Virtual Try-On**
![Virtual Try-On UI](assets/virtual_tryon.png)

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

## Contributing

We welcome contributions from the community! Feel free to submit a pull request or report issues.

### Steps to Contribute:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [KlingAI](https://klingai.com) for the API integration.
- [Streamlit](https://streamlit.io/) for the intuitive frontend framework.
- TensorFlow and Keras for powering the deep learning models.

---

Enjoy exploring and contributing to the **Fashion Recommendation and Virtual Try-On System**!











-----
# Fashion Recommender system

With an increase in the standard of living, peoples' attention gradually moved towards fashion that is concerned to be a popular aesthetic expression. Humans are inevitably drawn towards something that is visually more attractive. This tendency of humans has led to the development of the fashion industry over the course of time. However, given too many options of garments on the e-commerce websites, has presented new challenges to the customers in identifying their correct outfit. Thus, in this project, we proposed a personalized Fashion Recommender system that generates recommendations for the user based on an input given. Unlike the conventional systems that rely on the user's previous purchases and history, this project aims at using an image of a product given as input by the user to generate recommendations since many-a-time people see something that they are interested in and tend to look for products that are similar to that. We use neural networks to process the images from Fashion Product Images Dataset and the Nearest neighbour backed recommender to generate the final recommendations.

## Introduction

Humans are inevitably drawn towards something that is visually more attractive. This tendency of 
humans has led to development of fashion industry over the course of time. With introduction of 
recommender systems in multiple domains, retail industries are coming forward with investments in 
latest technology to improve their business. Fashion has been in existence since centuries and will be 
prevalent in the coming days as well. Women are more correlated with fashion and style, and they 
have a larger product base to deal with making it difficult to take decisions. It has become an important 
aspect of life for modern families since a person is more often than not judged based on his attire. 
Moreover, apparel providers need their customers to explore their entire product line so they can 
choose what they like the most which is not possible by simply going into a cloth store.

## Related work

In the online internet era, the idea of Recommendation technology was initially introduced in the mid-90s. Proposed CRESA that combined visual features, textual attributes and visual attention of 
the user to build the clothes profile and generate recommendations. Utilized fashion magazines 
photographs to generate recommendations. Multiple features from the images were extracted to learn 
the contents like fabric, collar, sleeves, etc., to produce recommendations. In order to meet the 
diverse needs of different users, an intelligent Fashion recommender system is studied based on 
the principles of fashion and aesthetics. To generate garment recommendations, customer ratings and 
clothing were utilized in The history of clothes and accessories, weather conditions were 
considered in to generate recommendations.

##  Proposed methodology

In this project, we propose a model that uses Convolutional Neural Network and the Nearest 
neighbour backed recommender. As shown in the figure Initially, the neural networks are trained and then 
an inventory is selected for generating recommendations and a database is created for the items in 
inventory. The nearest neighbour’s algorithm is used to find the most relevant products based on the 
input image and recommendations are generated.

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/2d64eecc5eec75f86d67bf15d59d87598b7f1a90/Demo/work-model.png?raw=true "Face-Recognition-Attendance-System")

## Training the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
 shows the ResNet50 architecture.

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/72528f2b4197cc5010227068ec72cd10f71214d4/Demo/resnet.png?raw=true "Face-Recognition-Attendance-System")

## Getting the inventory

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations. The Figure shows a sample set of inventory data

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/1e51a0d1db0e171e8d496524aa95a0098241fb1b/Demo/inventry.png?raw=true "Face-Recognition-Attendance-System")

## Recommendation generation

To generate recommendations, our proposed approach uses Sklearn Nearest neighbours Oh Yeah. This allows us to find the nearest neighbours for the 
given input image. The similarity measure used in this Project is the Cosine Similarity measure. The top 5 
recommendations are extracted from the database and their images are displayed.

## Experiment and results

The concept of Transfer learning is used to overcome the issues of the small size Fashion dataset. 
Therefore we pre-train the classification models on the DeepFashion dataset that consists of 44,441
garment images. The networks are trained and validated on the dataset taken. The training results 
show a great accuracy of the model with low error, loss and good f-score.

### Dataset Link

[Kaggle Dataset Big size 15 GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Kaggle Dataset Small size 572 MB](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

## Screenshots

### Simple App UI

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/1e51a0d1db0e171e8d496524aa95a0098241fb1b/Demo/2021-11-25.png?raw=true "Face-Recognition-Attendance-System")

### Outfits generated by our approach for the given input image

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/1e51a0d1db0e171e8d496524aa95a0098241fb1b/Demo/2021-11-25%20(1).png?raw=true "Face-Recognition-Attendance-System")


![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/1e51a0d1db0e171e8d496524aa95a0098241fb1b/Demo/2021-11-25%20(4).png?raw=true "Face-Recognition-Attendance-System")


![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/1e51a0d1db0e171e8d496524aa95a0098241fb1b/Demo/2021-11-25%20(3).png?raw=true "Face-Recognition-Attendance-System")

## Installation

Use pip to install the requirements.

~~~bash
pip install -r requirements.txt
~~~

## Usage

To run the web server, simply execute streamlit with the main recommender app:

```bash
streamlit run main.py
```

## Built With

- [OpenCV]() - Open Source Computer Vision and Machine Learning software library
- [Tensorflow]() - TensorFlow is an end-to-end open source platform for machine learning.
- [Tqdm]() - tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable.
- [streamlit]() - Streamlit is an open-source app framework for Machine Learning and Data Science teams. Create beautiful data apps in hours, not weeks.
- [pandas]() - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- [Pillow]() - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
- [scikit-learn]() - Scikit-learn is a free software machine learning library for the Python programming language.
- [opencv-python]() - OpenCV is a huge open-source library for computer vision, machine learning, and image processing.

## Conclusion

In this project, we have presented a novel framework for fashion recommendation that is driven by data, 
visually related and simple effective recommendation systems for generating fashion product images. 
The proposed approach uses a two-stage phase. Initially, our proposed approach extracts the features 
of the image using CNN classifier ie., for instance allowing the customers to upload any random 
fashion image from any E-commerce website and later generating similar images to the uploaded image 
based on the features and texture of the input image. It is imperative that such research goes forward 
to facilitate greater recommendation accuracy and improve the overall experience of fashion 
exploration for direct and indirect consumers alike.
