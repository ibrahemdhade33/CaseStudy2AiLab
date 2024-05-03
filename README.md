![University Logo](Birzeit-logo.jpg)

# Electrical and Computer Engineering Department

## Case Study#2 Report

### Submitted by:
**Name:** Ibraheem Duhaidi  
**University Number:** 1190283

### Submitted to:
**Doctor Name:** Yazan AbuFarha  

### Date:
**Submission Date:** 3/5/2024

---
## Abstract

This case study evaluates the performance of two classification techniques: Support Vector Machines (SVM) and Multilayer Perceptrons (MLP). Through experimental analysis on various datasets, we compare the effectiveness of each model in terms of accuracy and efficiency. The study aims to provide insights into the strengths and weaknesses of SVM and MLP, helping to guide the selection of appropriate classification methods for different data science applications.

---

## Introduction

In the realm of machine learning, classification tasks are pivotal for translating data into actionable insights. This study focuses on two robust classification techniques: Support Vector Machines (SVM) and Multilayer Perceptrons (MLP). SVM is renowned for its capacity to handle linear and non-linear boundaries through the use of kernels, while MLP, a type of neural network, excels in capturing complex patterns in data through multiple layers of processing.

Optimizing these models requires careful tuning of hyperparameters, for which Grid Search is a systematic approach. It searches exhaustively over a specified parameter space, enabling the identification of the most effective combinations for model performance.

Furthermore, dealing with high-dimensional data effectively necessitates dimensionality reduction techniques such as Principal Component Analysis (PCA). PCA reduces the dimensionality of data by transforming it into a new set of variables that are fewer in number but still capture most of the data's variability, which is essential for enhancing the computational efficiency and effectiveness of machine learning models.

This study conducts a comparative analysis to assess the performance of SVM and MLP in a controlled environment, leveraging Grid Search for optimization and PCA for dimensionality reduction, aiming to elucidate their practical utilities and limitations in handling diverse datasets.

---

## Procedure

This section outlines the methodology employed in evaluating the performance of Support Vector Machines (SVM) and Multilayer Perceptrons (MLP) classifiers. The analysis is divided into two parts: first, applying each classifier to individual labels; second, exploring the impact of combining labels on classifier performance.

### Part 1: Classification of Individual Labels

In the first part of the study, SVM and MLP classifiers were applied separately to three distinct labels within the dataset: Currency, Denomination, and Orientation. Each label was treated as a separate classification task with its own set of procedures:

1. **Data Preprocessing**: Standardize the features to ensure uniformity and improve classifier performance.
2. **Principal Component Analysis (PCA)**: PCA was applied to reduce data dimensionality for each label individually, with component counts tested at 50, 100, and 150 to observe the impact on performance.
3. **Grid Search for Hyperparameter Tuning**:
   - **SVM Parameters**: Conducted a grid search to find the optimal hyperparameters for the SVM classifier, including 'C', 'kernel', and 'gamma'.
   - **MLP Parameters**: Performed a similar grid search for the MLP classifier, exploring options for 'hidden_layer_sizes', 'activation', and 'learning_rate_init'.
4. **Model Training**: Each classifier was trained on the transformed data.
5. **Evaluation**: Performance metrics such as accuracy, precision, recall, and F1-score were used to evaluate each model.

### Part 2: Classification with Combined Labels

In the second part, labels were combined into a single classification target to evaluate the classifiers’ performance under more complex, multi-dimensional output conditions:

1. **Label Combination**: Currency, Denomination, and Orientation were concatenated to form a unique identifier for each class instance in the dataset.
2. **Data Preprocessing**: Standardization was performed on the combined data set.
3. **Principal Component Analysis (PCA)**: PCA was applied to the combined labels data, with component counts of 50, 100, and 150, to assess the impact of dimensionality reduction on classifier performance.
4. **Grid Search for Hyperparameter Tuning**:
   - **SVM Parameters**: Conducted a grid search to find the optimal hyperparameters for the SVM classifier, including 'C', 'kernel', and 'gamma'.
   - **MLP Parameters**: Performed a similar grid search for the MLP classifier, exploring options for 'hidden_layer_sizes', 'activation', and 'learning_rate_init'.
5. **Model Training**: Trained each classifier on the PCA-transformed data.
6. **Evaluation**: Evaluated using the same detailed metrics as in Part 1, providing a direct comparison of performance across individual and combined label scenarios.

This structured approach enables a thorough comparative analysis of SVM and MLP classifiers' capabilities, highlighting how each handles different complexities and dimensions of classification tasks.

---

## Results
### SVM and MLP Performance on Individual Labels with grid search

best training parameters for svm an mlp for each label

| Label         | SVM Best Parameters                           | MLP Best Parameters                                   |
|---------------|-----------------------------------------------|-------------------------------------------------------|
| Currency      | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} | {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001} |
| Denomination  | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} | {'activation': 'relu', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001} |
| Orientation   | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} | {'activation': 'relu', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001} |

#### Classification accuracy of Individual Labels

| Label         | Classifier | Accuracy |
|---------------|------------|----------|
| Currency      | SVM        | 98%      |
|               | MLP        | 98%      |
| Denomination  | SVM        | 95%      |
|               | MLP        | 95%      |
| Orientation   | SVM        | 96%      |
|               | MLP        | 95%      |

### SVM and MLP Performance on Individual Labels with PCA

for testing the effect of the pca, we used 50, 100, and 150 components, and  the best parameters from the grid search was taken for each label
and this results was obtained
##### Currency

| PCA Components | SVM Accuracy | MLP Accuracy |
|----------------|--------------|--------------|
| 50             | 97%          | 95%          |
| 100            | 98%          | 97%          |
| 150            | 98%          | 98%          |

##### Denomination

| PCA Components | SVM Accuracy | MLP Accuracy |
|----------------|--------------|--------------|
| 50             | 94%          | 92%          |
| 100            | 95%          | 94%          |
| 150            | 95%          | 95%          |

##### Orientation

| PCA Components | SVM Accuracy | MLP Accuracy |
|----------------|--------------|--------------|
| 50             | 95%          | 93%          |
| 100            | 96%          | 95%          |
| 150            | 96%          | 95%          |



### SVM and MLP Performance on Combined Labels with grid search

| Classifier | Best Parameters                                                   | Accuracy |
|------------|-------------------------------------------------------------------|----------|
| SVM        | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                      | 95%      |
| MLP        | {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001} | 95%      |


### SVM and MLP classifiers achieved comparable accuracy levels when applied to the combined labels dataset on PCA 
| Classifier | PCA Components | Accuracy |
|------------|----------------|----------|
| SVM        | 50             | 94%      |
| SVM        | 100            | 95%      |
| SVM        | 150            | 95%      |
| MLP        | 50             | 93%      |
| MLP        | 100            | 94%      |
| MLP        | 150            | 95%      |

### Discussion

In our experiments, we compared the performance of Support Vector Machines (SVM) and Multilayer Perceptrons (MLP) under various configurations, employing Grid Search, Principal Component Analysis (PCA), and both combined and individual labeling strategies. The results across all experiments were notably consistent, highlighting several key insights:

- **Impact of PCA**: The application of PCA for dimensionality reduction proved effective. We achieved comparable performance levels without utilizing all the principal components, demonstrating PCA's utility in enhancing model efficiency.
  
- **Model Performance with PCA**: Both SVM and MLP showed improved performance when PCA was integrated, suggesting that reducing the number of features can lead to more efficient model training without sacrificing accuracy.
  
- **Label Strategy Comparison**: When examining the effects of different labeling strategies, the performance between individual labels was similar. However, models trained with currency labels consistently outperformed others, indicating a possible advantage in using these labels for our specific dataset.
  
- **Comparison Between SVM and MLP**: Although the overall performance of SVM and MLP was close, SVM consistently exhibited slightly better accuracy than MLP. This suggests that for this particular problem and dataset, SVM may be more effective.

- **Role of Grid Search**: The use of Grid Search was crucial in our experiments. It helped in finding the best parameters for our models, which significantly contributed to optimizing their performance under varying configurations and conditions.

### Conclusion

The experimental results indicate that both SVM and MLP can achieve satisfactory and comparable results when combined with PCA, which effectively reduces feature space complexity. Additionally, using currency labels provides a slight edge in performance, particularly with the SVM model, which consistently outperformed the MLP. The implementation of Grid Search was essential for tuning the models to achieve optimal parameter settings, further enhancing their effectiveness. These findings underscore the importance of choosing the right dimensionality reduction technique and labeling strategy to enhance model performance in practical applications. This experiment contributes valuable insights into optimizing machine learning workflows for similar tasks.

