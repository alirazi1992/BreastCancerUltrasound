# Machine Learning Approaches for Ultrasound-Based Breast Cancer Detection

**Data Scientist:** Ali Razi  
**Contact:** ali.razi9292@gmail.com

## Objective

The objective of this project is to leverage machine learning and artificial intelligence techniques to improve the accuracy and automation of breast cancer detection using ultrasound images. Early breast cancer detection is crucial for patient outcomes, and employing AI and ML can assist radiologists, reduce false-positive results, and aid in data analysis and research. This project aims to develop models capable of detecting breast cancer at an early stage, potentially before it metastasizes.

## My Capstone Project

### Notebooks

#### Notebook #1: Machine-learning approaches for ultrasound-based breast cancer detection

- **Introduction**
- **Dataset Description**
- **Exploratory Data Analysis**
  - A. Data Cleaning
  - B. Feature Exploration
  - C. Feature Processing
- **CNN Model for Breast Cancer Detection**

##### Introduction

Breast cancer is a major health concern, particularly among women. To address this, artificial intelligence (AI) and machine learning (ML) techniques have gained significant attention for their ability to automate image analysis and improve diagnostic accuracy. In the context of breast cancer screening, AI can contribute to early detection and more precise interpretations of breast ultrasound exams. While mammography remains the gold standard for breast cancer diagnosis, it is not always reliable, especially for cases involving dense breasts. This project explores the potential of AI and ML in improving breast cancer detection using ultrasound images.

##### Dataset Description

The dataset used in this project comprises breast ultrasound images categorized into two classes: benign and malignant. Additionally, the dataset is divided into training and validation sets. The data source is Kaggle, and you can find the dataset [here](https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer).

The dataset primarily consists of PNG image files, each representing an individual breast ultrasound image. Before feeding these images into the machine learning model, they will undergo preprocessing to convert them into numerical representations. The dataset also includes corresponding labels (0 for benign and 1 for malignant) to indicate the ground truth for each image. Various versions of each image are provided to enhance model robustness.

##### Exploratory Data Analysis

###### A. Data Cleaning

Data cleaning is a critical step in preparing the dataset for analysis. After splitting the dataset into training, validation, and test sets, it's essential to perform data sanity checks and modifications. These checks ensure data validity, accuracy, completeness, consistency, and uniformity.

To maintain data quality and model accuracy, data cleaning includes:
1. Resizing images to a uniform size for consistency.
2. Handling corrupted or unreadable images.
3. Verifying correct labeling of images.
4. Augmenting data through techniques like rotation or flipping.

###### B. Feature Exploration

Feature exploration in this project focuses on understanding the characteristics of ultrasound images. As the data primarily consists of PNG images, feature exploration involves visualizing and analyzing the images themselves. Key steps include:

1. Visualizing Sample Images: Displaying random samples to understand data variability.
2. Examining Class Distribution: Ensuring a balanced representation of benign and malignant cases.
3. Checking Image Size and Resolution: Ensuring uniform dimensions for compatibility.
4. Determining Image Channels: Grayscale vs. RGB.
5. Data Augmentation: Visualizing augmented images.
6. Sample Image Preprocessing: Visualizing preprocessing transformations.
7. Identifying Image Quality Issues: Inspecting images for artifacts or noise.
8. Visualizing Class-Specific Features: Exploring distinguishing features.
9. Feature Visualization with CNN: Understanding patterns learned by the model.
10. Denoising: Reducing noise in ultrasound images.
11. Balancing: Ensuring equal representation of image classes.

###### C. Feature Processing

Feature processing involves preparing data for specific modeling algorithms, enhancing predictive accuracy, and promoting generalization. Given that the data consists of PNG images, feature processing takes a different approach compared to traditional tabular data. Key techniques include:

1. Image Resizing: Ensuring uniform image sizes for model compatibility.
2. Normalization: Scaling pixel values to a common range.
3. Color Space Conversion: Converting RGB images to grayscale when color information is unnecessary.
4. Data Augmentation: Creating additional training samples through transformations.
5. Feature Extraction with CNN: Using Convolutional Neural Networks for feature extraction.

##### CNN Model for Breast Cancer Detection

This project focuses on the implementation of Convolutional Neural Networks (CNNs) for breast cancer detection. CNNs are well-suited for image classification tasks and have shown remarkable performance in various medical imaging applications. The CNN model is designed and trained to classify breast ultrasound images into benign and malignant categories.

### MobileNetV3 and Lime Integration

MobileNetV3, a pre-trained neural network architecture, is integrated into the project for feature extraction and classification tasks. MobileNetV3 is known for its efficiency and accuracy in image-related tasks. Additionally, the Lime technique is used to provide explainability to the MobileNetV3 model's predictions, allowing us to understand its decision-making process at the local level.

### Notebook #2: Model Evaluation and Interpretation

This notebook delves into the evaluation and interpretation of the CNN and MobileNetV3 models for breast cancer detection. Key steps include:

- Model Training: Training CNN and MobileNetV3 models.
- Model Evaluation: Assessing model performance using relevant metrics such as accuracy, precision, recall, F1-score, and AUC.
- Interpretation with Lime: Using the Lime technique to interpret model predictions and understand feature importance.
- Conclusion and Insights: Summarizing project outcomes, strengths, limitations, and areas for improvement.

## Data Environment

To recreate the data environment used for this project, you can create a conda environment with the following packages:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- jupyter
- jupyterlab
  
## Other Files

- **Final Report**
- **Presentation**
- **Column_names**: A list of variables in the original dataframe before cleaning and preprocessing.
