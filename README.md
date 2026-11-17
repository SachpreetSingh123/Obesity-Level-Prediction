# Obesity Level Prediction Using Machine Learning

## Overview

This project focuses on predicting obesity levels using Machine Learning techniques based on an individual's eating habits, physical condition, and lifestyle factors. The objective is to analyze health-related data and identify the most effective machine learning model for obesity prediction.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, classification, regression, clustering, and model evaluation using multiple machine learning algorithms.

---

## Dataset Information

Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition

- Total Records: 2,111
- Total Features: 17
- Target Variable: NObeyesdad (Obesity Level)

### Features Used

- Age
- Height
- Weight
- Gender
- Family History of Overweight
- Frequency of Vegetable Consumption
- Number of Main Meals
- Water Intake
- Physical Activity Frequency
- Technology Usage Time
- Smoking Habit
- Transportation Mode
- Alcohol Consumption
- High-Calorie Food Consumption
- Calorie Consumption Monitoring
- Food Consumption Between Meals

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Scaling
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Prediction and Analysis

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Label Encoding for categorical variables
- Feature Scaling using StandardScaler
- Train-Test Split (80:20)
- Data Cleaning and Preparation

---

## Machine Learning Models

### Classification Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor
- Support Vector Regressor (SVR)

### Clustering Models

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

---

## Model Performance

### Classification Accuracy

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 87.47% |
| Decision Tree | 93.14% |
| Random Forest | 94.80% |
| KNN | 82.03% |
| SVM | 88.89% |

### Best Classification Model

**Random Forest Classifier**

Accuracy: **94.80%**

Reasons for Strong Performance:
- Reduced overfitting through ensemble learning
- Better generalization on unseen data
- Effective handling of non-linear relationships

---

## Regression Results

| Model | MSE | R² Score |
|---------|---------|---------|
| Linear Regression | 2.72 | 0.26 |
| Decision Tree Regressor | 0.87 | 0.77 |
| Random Forest Regressor | 0.40 | 0.89 |
| KNN Regressor | 1.52 | 0.59 |
| SVR | 1.45 | 0.61 |

### Best Regression Model

**Random Forest Regressor**

- Lowest Mean Squared Error
- Highest R² Score (0.89)

---

## Clustering Results

| Algorithm | Silhouette Score |
|------------|------------------|
| K-Means | 0.119 |
| Hierarchical Clustering | 0.145 |
| DBSCAN | -0.219 |

### Best Clustering Method

**K-Means Clustering**

Successfully identified meaningful clusters within the dataset.

---

## Feature Importance

The most influential features affecting obesity prediction were:

1. Weight
2. Physical Activity Frequency
3. Eating Habits
4. Water Intake

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Key Insights

- Weight is the most significant factor influencing obesity levels.
- Physical activity has a strong impact on obesity prediction.
- Lifestyle habits play a crucial role in obesity classification.
- Ensemble learning methods outperform individual machine learning models.

---

## Future Improvements

- Hyperparameter Optimization
- Deep Learning-Based Prediction Models
- Real-Time Prediction System
- Web Application Deployment
- Integration with Healthcare and Wearable Device Data

---

## Project Structure
