#!/usr/bin/env python
# coding: utf-8

# # Data Mining Project
# ## Obesity Level Prediction using Machine Learning
# 
# **Name:** Sachpreet Singh  
# 

# In[2]:


# Importing required libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


# # Load Dataset

# In[4]:


df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

print("Shape of Dataset:", df.shape)
df.head()


# # Dataset Info

# In[6]:


df.info()
df.describe()


# # Data Visualization

# In[7]:


plt.figure(figsize=(6,4))
sns.countplot(x=df["NObeyesdad"])
plt.title("Distribution of Obesity Levels")
plt.xticks(rotation=45)
plt.show()


# # Data Preprocessing

# In[9]:


# Encoding categorical variables
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Splitting features and target
X = df.drop("NObeyesdad", axis=1)
# Encode the target text labels into numbers
le = LabelEncoder()
y = le.fit_transform(df['NObeyesdad'])

# Feature scaling
# Convert categorical text columns into numeric variables first
X_encoded = pd.get_dummies(X, drop_first=True)

# Feature scaling on the encoded data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# # Classification Section 

# # 1. Logistic Regression

# In[10]:


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# # 2. Decision Tree

# In[11]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# # 3. Random Forest

# In[12]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# # 4. KNN

# In[14]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# # 5. SVM

# In[15]:


svm = SVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


# # 6. Accuracy Comparison Graph

# In[16]:


acc = {
    "LR": accuracy_score(y_test, y_pred_lr),
    "DT": accuracy_score(y_test, y_pred_dt),
    "RF": accuracy_score(y_test, y_pred_rf),
    "KNN": accuracy_score(y_test, y_pred_knn),
    "SVM": accuracy_score(y_test, y_pred_svm),
}

plt.bar(acc.keys(), acc.values())
plt.title("Classification Accuracy Comparison")
plt.show()


# # REGRESSION SECTION

# # 1. Regression Models

# In[17]:


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR()
}

for name, model in models.items():
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)

    print("\n", name)
    print("MSE:", mean_squared_error(y_test_r, y_pred))
    print("R2:", r2_score(y_test_r, y_pred))


# # CLUSTERING SECTION

# # 1. Clustering

# In[19]:


kmeans = KMeans(n_clusters=5)
k_labels = kmeans.fit_predict(X_scaled)
print("KMeans Silhouette:", silhouette_score(X_scaled, k_labels))

hc = AgglomerativeClustering(n_clusters=5)
h_labels = hc.fit_predict(X_scaled)
print("Hierarchical Silhouette:", silhouette_score(X_scaled, h_labels))

db = DBSCAN()
d_labels = db.fit_predict(X_scaled)

if len(set(d_labels)) > 1:
    print("DBSCAN Silhouette:", silhouette_score(X_scaled, d_labels))
else:
    print("DBSCAN failed to form clusters")


# # 2. Feature Importance

# In[20]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# Get the feature importances from the Random Forest model
importance = rf.feature_importances_

# Clear any previous figures to avoid overlapping plots
plt.figure()

# Plot the horizontal bar chart
plt.barh(X_encoded.columns, importance)
plt.title("Feature Importance")

# FIX: Tell Streamlit to render the matplotlib figure
st.pyplot(plt)



