# Machine-learning approaches for ultrasound-based breast cancer detection

Data Scienttist: Ali Razi 
Contact: ali.razi9292@gmail.com

#### Objective 

Using machine learning in ultrasound breasts can be used as a tool to improve accuracy and automate image analysis. More importantly, it can detect cancer in an early stage before it metastasis. Employing AI and ML can assist radiologists by reducing false-positive results in the intrpretation of breast ultrasound exams and aid in data analysis and research.
Employing ML can be tool to measuring mass of tumore in breast cancer and help paitients for early diagnoitic process.

# My Capstone Project

#### Notebooks

###### Notebook #1: Machine-learning approaches for ultrasound-based breast cancer detection

    - Introduction 
    - Dataset Description 
    - Exploratory Data Analysis 

        A. Data Cleaning 
        B. Feature Exploration 
        C. Feature Processing 
 
    - Advanced Statistic Anlysis 


   ##### - Introdcution 

 - Artificial intelligence (AI) is gaining extensively attention due to its remarkable performance in image-recognition tasks and its increasing utilizing in breast ultrasound. AI can conduct a quantitive assessment by recognizing imaging information automatically and make more precise and consistent imaging diagnosis. Breast cancer is the most commonly cancer among women which is severely threatening their health. By early screening of which is closely related to the prognosis of patients. Aa a result by employing the AI in breast cancer screening and detection holds immense importance, which can not only save time for radiologist , but also make up for experience and skill deficiency on some beginners. Additionally, mammography is the gold standard for the breast cancer diagnosis, but it is not reliable and accurate for all cases, specifically for cases with dense breasts.

   ##### - Dataset Descrioption 

   >> Breast Ultrasound Images Dataset is reviewing the medical images of breast cancer using an ultrasound scan. This database is categorized into three classes: 1. Bennign and 2.Malignant images. Also, it has been classified into a train set and a validation set. Data has been obtained from the kaggle database: https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer .

   >> The dataset is split into two parts: the training set and the validation set. The training set will be used to train the machine learning model, while the validation set will be used to evaluate its performance. Also, the dataset contains PNG images files, each representing an individual breast ultrasound image. Each image will be preprocessed and transformed into a numerical representation, usually a matrix or array of pixel values, before feeding it into the machine learning model.

   >> The dataset will also include corresponding labels (0 for benign and 1 for malignant) indicating the ground truth for each image.

   >> Additionally, each sample data(image) is provided in certain formats of: 
   1.rotated1-rotated2.png, 2.rotated1-sharpened.png, 3.rotated2-rotated1.png, 
   4.rotated2-rotated2.png, 5.rotated2-sharpened.png, 6.rotated2.jpg, 7.rotated2.png,
   8.sharpened-rotated1.png, 9.sharpened-rotated2.png, 10.sharpened-sharpened.png, 
   11.sharpened.jpg, 12.sharpened.png.

 ##### - Exploratory Data Analysis  

 ### A. Data Cleaning


 For having better vision and perspective of data it woulb better to do sanity check and modify data. There are several key steps that for machine learning sanity chehck and data modification befire EDA is needed: I. Validity, II.Accuracy, III.Completness, IV. Consistensy, V. Uniformity 

 
 - To ensure quality and database and the accuracy of the model. Multiple steps such as:
 1. Resize images to a uniform size to ensure consistency.
 2. Handling any corrupted or unreadable images.
 3. Verifying that all images are correctly labeled with the corresponding class (benign or malignant).
 4. Augmenting the data through techniques like rotation or flipping to increase the training set size.

#### B. Feature Exploration

- For investigating and analyzing the individual features present in the dataset as the data in this project are in PNG format feature exploration involves visualizing and understanding the characteristics of the images themselves rather than examining numerical statistics or distributions. This can contain some key steps such as: 

 1. Visualizing Sample Images: Display a random sample of PNG images from the dataset. This will give an initial sense of the data and help to understand the variability and quality of the images.

 2. Class Distribution: Examine the distribution of different classes (benign and malignant) in the dataset. Ensure that it has a balanced representation of both classes to avoid any class imbalance issues during modeling.

 3. Image Size and Resolution: Check if all images have the same dimensions and resolution. Ensuring uniform image sizes is essential for compatibility with machine learning models.
 
 4. Image Channels: Determine whether the images are grayscale (single-channel) or RGB (three-channel). Grayscale images have one channel, while RGB images have three channels representing red, green, and blue.

 5. Data Augmentation: By applying data augmentation techniques during data preprocessing, visualize some augmented images to understand the variations introduced in the training set.

 6. Sample Image Preprocessing: Apply any preprocessing transformations performed during data preparation to a few sample images and visualize the results.

 7. Identifying Image Quality Issues: Inspect images for any quality issues, artifacts, or noise that may impact model performance.

 8. Visualizing Class-Specific Features: Explore whether any unique visual features distinguish benign images from malignant images.

 9. Feature Visualization with CNN: it can visualize the feature maps produced by intermediate layers of the network to understand what patterns the model is learning from the images.

### C.Feature Processing

- To represent the data more suitably for the specific modeling algorithm, leading to better predictive accuracy and generalization following techniques can be helpful:(It should be mentioned that te data here is in the foramt od PNG images, therefore,feature processing, or feature engineering, takes a different approach compared to traditional tabular data. For image data, feature processing involves transforming raw image pixels into meaningful and representative features that can be fed into machine learning models)

 1. Image Resizing:
 Resize all images to a uniform size to ensure compatibility with machine learning models. This step is necessary when images have different dimensions.
 2. Normalization:
 Normalize the pixel values of the images to a common range (e.g., [0, 1] or [-1, 1]). This ensures that the features have a similar scale, which can improve model convergence. 
 3. Color Space Conversion:
 Convert RGB images to grayscale if applicable, especially if color information is not crucial for the task at hand. Grayscale images reduce computational complexity and memory requirements. 
 4. Data Augmentation:
 Apply data augmentation techniques to create additional training samples by performing transformations such as rotation, flipping, zooming, and shifting. Data augmentation helps improve model generalization and robustness. 
 5. Feature Extraction with CNN:
 Use Convolutional Neural Networks (CNNs) as feature extractors. You can remove the fully connected layers from a pre-trained CNN and use the intermediate layers as features. These learned features can then be fed into traditional machine learning models like SVM or Random Forest. 
 6. Transfer Learning:
 Utilize transfer learning with pre-trained CNN models. Fine-tune the weights of a pre-trained CNN on your specific task or use it as a feature extractor.
 7. Histogram of Oriented Gradients (HOG):
 Extract HOG features from the images, which represent the distribution of gradient orientations in different regions of the image. HOG features are commonly used in object detection tasks. 
 8. Local Binary Patterns (LBP):
 Compute Local Binary Patterns from the grayscale images to capture texture information. LBP features are often used in texture analysis and facial recognition tasks.
 9. Edge Detection:
 Perform edge detection on the images to extract edges and contours, which can be useful in certain image analysis tasks.
 10. Deep Feature Extraction (with autoencoders ):
 Use unsupervised deep learning techniques like autoencoders to extract meaningful features from the images.   

#### Advanved Statistic Analysis

- To apply sophisticated statistical techniques to gain deeper insights into the data, make predictions, and draw more meaningful conclusions from the data, some advanced statistical analysis techniques consider:
1. Convolutional Neural Networks (CNN):
Utilize pre-trained CNN models (e.g., VGG, ResNet, Inception) to extract high-level features from the images. Remove the fully connected layers of the pre-trained model and use the intermediate convolutional layers as feature vectors.
2. Transfer Learning with Fine-tuning:
Fine-tune a pre-trained CNN on your specific breast cancer ultrasound image dataset. Retrain the last few layers of the CNN on your data while keeping the early layers frozen to preserve learned features.
3. Feature Visualization from CNN:
Visualize the learned features in intermediate layers of the CNN to gain insights into what patterns and features the model is capturing from the images.
4. Image Segmentation with Deep Learning:
Use deep learning-based image segmentation techniques to segment breast tumor regions in the ultrasound images, allowing for more targeted analysis of specific areas.
5. Gaussian Mixture Models (GMM):
Apply Gaussian Mixture Models to model the distribution of image features and identify clusters or patterns in the data.
6. Principal Component Analysis (PCA) for Image Features:
Perform PCA on the extracted image features to reduce dimensionality and identify the most important components that capture the variability in the data.
7. Variational Autoencoders (VAEs):
Use VAEs to learn a low-dimensional representation of the images, enabling data compression and potentially generating new, realistic images.
8. Generative Adversarial Networks (GANs) for Data Augmentation:
Utilize GANs to generate synthetic images that can augment of dataset, effectively increasing the size of the training set and improving model generalization.

###### Notebook #2: Modeling Machine-learning approaches for ultrasound-based breast cancer detection

- The primary goal of the project is to leverage machine learning algorithms to build effective models that can detect breast cancer from ultrasound images. Each step is carefully tailored to the topic of machine learning for breast cancer detection, focusing on the unique challenges posed by ultrasound images and PNG format. The provided information steps are common in machine learning project, particularly in the context of image data analysis and classification tasks, and its application to the specific domain of ultrasound breast cancer detection.

    - Baseline Model for Breast Cancer Detection:
        Begin the project by creating a baseline model using machine learning techniques to detect breast cancer from ultrasound images.
   
    - Logistic Regression for Breast Cancer Classification:
        Implement logistic regression as an initial model for classifying ultrasound breast cancer images.
  
    - Handling Class Imbalance in Breast Cancer Data:
        Address the class imbalance issue in the breast cancer dataset using appropriate techniques to ensure a balanced representation of benign and malignant cases.
  
    - Feature Selection for Breast Cancer Detection:
        Explore feature selection methods specifically tailored for ultrasound breast cancer images, extracting relevant features to improve model performance.
      
    - Hyperparameter Optimization and Cross-Validation:
       Optimize hyperparameters and utilize cross-validation techniques to fine-tune models, including logistic regression, decision trees, random forest, and XGBoost, for better predictive accuracy.

    - Customized Image Pipelines for Breast Cancer Detection:
        Create specialized pipelines for image preprocessing, feature extraction, and model training, tailored to the unique characteristics of ultrasound breast cancer images.

    - Final Model Evaluation for Breast Cancer Detection:
       Evaluate the performance of the final models using appropriate metrics, such as accuracy, precision, recall, F1-score, and Area Under the Curve (AUC), ensuring effective assessment of breast cancer detection capabilities.

    - Conclusion and Insights for Ultrasound Breast Cancer Detection:
      Summarize the outcomes of the machine learning project, including the strengths and limitations of each model for ultrasound breast cancer detection.
      Provide insights and potential areas for further improvement in the application of machine learning to breast cancer diagnosis. 


#### Data Environment 

>> conda create -n capstone_AliRazi python=3.9 numpy pandas matplotlib seaborn tensorflow jupyter jupyterlab


#### other files

    - Final Report
    - Presentation
    - Column_names: the list of the varaibles in the original dataframe befor ecleaning and preprocessing