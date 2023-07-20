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