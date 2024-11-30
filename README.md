# Exploring-Pandas

# Cardiovascular Disease Prediction Using Machine Learning 

This project focuses on analyzing and comparing the performance of three machine learning models—Linear Regression, K-Nearest Neighbors (KNN), and Logistic Regression—using a cardiovascular dataset.

## Table of Contents 
 
1. [Introduction](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#introduction)
 
2. [Project Structure](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#project-structure)
 
3. [Dataset Preparation](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#dataset-preparation)
 
4. [Model Implementation](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#model-implementation)  
    - [Linear Regression](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#linear-regression)
 
    - [K-Nearest Neighbors (KNN)](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#k-nearest-neighbors-knn)
 
    - [Logistic Regression](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#logistic-regression)
 
5. [Conclusion](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#conclusion)
 
6. [References](https://chatgpt.com/c/674ae712-3014-800c-b65b-9ad7f94a1c18#references)


---


## Introduction 

The goal of this project was to evaluate the effectiveness of three machine learning models for predicting cardiovascular disease. By training these models with a dataset, we analyzed their performance using accuracy scores.


---


## Project Structure 
 
1. **Dataset Preparation**  
  - Load and inspect the dataset (`dataset.csv`).

  - Analyze and visualize the data using statistical tools and plots.
 
2. **Model Implementation** 
  - Train and test the dataset in a 7:3 ratio.

  - Use five different sets of input parameters to evaluate the models.
 
3. **Performance Evaluation** 
  - Measure accuracy scores to compare the models.


---


## Dataset Preparation 
 
- **Libraries Used** :
`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`
 
- **Steps** : 
  1. Load the dataset using `pandas`.

  2. Explore data statistics and visualize using plots.

  3. Preprocess the data if necessary (e.g., handling missing values, normalization).


---


## Model Implementation 

### Linear Regression 
 
- **Objective** : Predict target values based on selected input parameters.
 
- **Steps** : 
  1. Import the `LinearRegression` module.

  2. Split the dataset into training and testing sets.

  3. Train the model and make predictions.

  4. Evaluate the model using accuracy scores.

### K-Nearest Neighbors (KNN) 
 
- **Objective** : Classify the dataset based on proximity to neighbors.
 
- **Steps** : 
  1. Import the `KNeighborsClassifier` module.

  2. Configure the number of neighbors as a hyperparameter.

  3. Train the model and evaluate its performance on the test set.

### Logistic Regression 
 
- **Objective** : Perform binary classification to predict the presence of cardiovascular disease.
 
- **Steps** : 
  1. Import the `LogisticRegression` module.

  2. Train the model on selected parameters.

  3. Predict outcomes for the test dataset and calculate accuracy.


---


## Conclusion 
 
- **Results** : 
  - **KNN**  performed poorly, likely due to its sensitivity to data distribution and hyperparameter settings.
 
  - **Linear Regression**  and **Logistic Regression**  both performed well, with **Logistic Regression**  slightly outperforming due to its capability to handle binary classification tasks.
 
- **Takeaways** :
  - Logistic regression is an effective method for healthcare data analysis.

  - Proper algorithm selection and parameter tuning are crucial for achieving reliable results.


---


## References 
 
- [GitHub Repository](https://github.com/AnkitSinghGTHB/Exploring-Pandas)
 
- [Google Colaboratory Link (Accessible by VIT Mail)](https://colab.research.google.com/drive/1-XPZagIviLkiZ7vEHnBPIN2JDd8yy2wK?usp=sharing)
