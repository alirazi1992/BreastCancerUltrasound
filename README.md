# Machine-learning approaches for ultrasound-based breast cancer detection

Data Scienttist: Ali Razi 
Contact: ali.razi9292@gmail.com

#### Objective 

Using machine learning in ultrasound brest can be used as a tool to improve accuracy and automate image analysis.More importantly it can detect cacner in early stage before it was metastasis. Employing AI and ML can assits radiologsit by reducing false-positive results in intrpretation of breast ultrasound exams and aid in data analysis and research.

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



   ##### - Dataset Descrioption 

   >> Breast Ultrasound Images Dataset is reviewing the medical images of breast cancer that using ultrasound scan. This database is categorized into three classes: 1. Bennign and 2.Malignant images. Also, it has been classified into train set and validation set. Data has been obtained from kaggle database : https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer .

   >> The dataset is split into two parts: the training set and the validtion set. The training set will be used to train the machine learning model, while the validation set will be used to evaluate its performance. Also,the dataset contains of PNG images files, each representing an individual breast ultrasound image. Each image will be preprocessed and transformed into a numerical representation, usually a matrix or array of pixel values, before feeding it into the machine learning model.

   >> The dataset will also include corresponding labels (0 for benign and 1 for malignant) indicating the ground truth for each image.

   >> Additionally, the each sample data(image) provided in ceratin formats of : 1.rotated1-rotated2.png, 2.rotated1-sharpened.png, 
   3.rotated2-rotated1.png, 4.rotated2-rotated2.png, 5.rotated2-sharpened.png, 6.rotated2.jpg, 7.rotated2.png,8.sharpened-rotated1.png, 9.sharpened-rotated2.png,10.sharpened-sharpened.png,11.sharpened.jpg,12.sharpened.png.

 ##### - Exploratory Data Analysis  

 ### A. Data Cleaning
 
 - To ensure quality and database and the accuracy of the model. Multiple steps such as:
                            1.Resizing images to a uniform size to ensure consistency.
                            2.Handling any corrupted or unreadable images.
                            3.Verifying that all images are correctly labeled with the corresponding class (benign or malignant).
                            4.Augmenting the data through techniques like rotation or flipping to increase the training set size.

#### B. Feature Exploration

- For investigating and analyzing teh individual features presernt in the dataset as the data in this project are in PNG format feature exploration involves visualizing and understanding the characteristics of the images themselves rather than examining numerical statistics or distributions. This can conatin some key stps such as: 
                            1. Visualizing Sample Images: Display a random sample of PNG images from the dataset. This will give an initial sense of the data and help to understand the variability and quality of the images.

                            2. Class Distribution: Examine the distribution of different classes (benign and malignant) in the dataset. Ensure that it has a balanced representation of both classes to avoid any class imbalance issues during modeling.

                            3.Image Size and Resolution: Check if all images have the same dimensions and resolution. Ensuring uniform image sizes is essential for compatibility with machine learning models.
                           
                            4.Image Channels: Determine whether the images are grayscale (single-channel) or RGB (three-channel). Grayscale images have one channel, while RGB images have three channels representing red, green, and blue.

                            5.Data Augmentation : By applying data augmentation techniques during data preprocessing, visualize some augmented images to understand the variations introduced in the training set.

                            6. Sample Image Preprocessing: Apply any preprocessing transformations it  performed during data preparation to a few sample images and visualize the results.

                            7.Identifying Image Quality Issues: Inspect images for any quality issues, artifacts, or noise that may impact model performance.

                            8.Visualizing Class-Specific Features: Explore whether there are any unique visual features that distinguish benign images from malignant images.

                            9.Feature Visualization with CNN: it can visualize the feature maps produced by intermediate layers of the network to understand what patterns the model is learning from the images.

### C.Feature Processing

- To representing the data in a more suitable way for the specific modeling algorithm, leading to better predictive accuracy and generalization following techniques can be helpful:(It should be mentioend that te data here is in foramt od PNG images,therefore,feature processing, or feature engineering, takes a different approach compared to traditional tabular data. For image data, feature processing involves transforming raw image pixels into meaningful and representative features that can be fed into machine learning models)

                            1. Image Resizing:
                                     Resize all images to a uniform size to ensure compatibility with machine learning models. This step is necessary when images have different dimensions.
                            2.  Normalization:
                                     Normalize the pixel values of the images to a common range (e.g., [0, 1] or [-1, 1]). This ensures that the features have a similar scale, which can improve model convergence.  
                            3.Color Space Conversion:
                                     Convert RGB images to grayscale if applicable, especially if color information is not crucial for the task at hand. Grayscale images reduce computational complexity and memory requirements.   
                            4.Data Augmentation:
                                     Apply data augmentation techniques to create additional training samples by performing transformations such as rotation, flipping, zooming, and shifting. Data augmentation helps improve model generalization and robustness. 
                            5.Feature Extraction with CNN:
                                     Use Convolutional Neural Networks (CNNs) as feature extractors. You can remove the fully connected layers from a pre-trained CNN and use the intermediate layers as features. These learned features can then be fed into traditional machine learning models like SVM or Random Forest.                    
                            6.Transfer Learning:
                                     Utilize transfer learning with pre-trained CNN models. Fine-tune the weights of a pre-trained CNN on your specific task or use it as a feature extractor.
                            7.Histogram of Oriented Gradients (HOG):
                                     Extract HOG features from the images, which represent the distribution of gradient orientations in different regions of the image. HOG features are commonly used in object detection tasks.         
                            8.Local Binary Patterns (LBP):
                                     Compute Local Binary Patterns from the grayscale images to capture texture information. LBP features are often used in texture analysis and facial recognition tasks.
                            9. Edge Detection:
                                     Perform edge detection on the images to extract edges and contours, which can be useful in certain image analysis tasks.
                            10.Deep Feature Extraction (with autoencoders ):
                                     Use unsupervised deep learning techniques like autoencoders to extract meaningful features from the images.         

#### Advanved Statistic Analysis

- To applying sophisticated statistical techniques to gain deeper insights into the data, make predictions and draw more meaningful cocnlusions from the data, there are some advanced statistical analysis techniques that consider:



###### Notebook #2: Modeling Machine-learning approaches for ultrasound-based breast cancer detection


    - Baseline Model
        -Logistic Regression 
    - Class imbalance 
        - Downsampling /Imbalane learn 
    - Feature Selection
        - Varaince Threshold
        - K-best
        - PCA
    - Hiperparameter Optimization and Cross Validation 
        - Logistic Regression 
        - Descision Trees
        - Random forest 
        - XGBoost 
    - Pipelines
        - Logistic Regression 
        - Random forest
        - XGBoost
    - Final Model Evaluation 
        - Scores
        - Area Under the Cure
    -Conclusion 

#### Data resource 

The dataset used in this project was originally collected and published by      in their paper. I will be using this dataset as astarting point for my own data science project, with the aim of furthur exploring the relationships and pattern sthat were observed in the original study. 
Refrence :

#### Data Environment 

>> conda create -n er_predictor python=3.9 numpy pandas matplotlib seaborn scikit-learn=0.24.1 jupyter jupyterlab
>> conda activate er_predictor

Additional libraries:
>> conda install -c conda-forge missingno
>> conda install -c conda-forge imbalanced-learn
>> conda install -c conda-forge xgboost=1.1.1 mlxtend

Kernel:
>> ipython kernel install --name "er_predictor" --user

#### other files

    - Final Report
    - Presentation
    - Column_names: the list of the varaibles in the original dataframe befor ecleaning and preprocessing