# Customer-Churn-Prediction


## Overview
This project is a **customer churn prediction** model using **Random Forest Classifier** in Python. The goal is to analyze customer behavior and predict whether a customer will churn based on various features.

## Dataset
- The dataset contains customer details and churn status.
- It consists of two files: `train_data.csv` (customer information) and `churn_data.csv` (churn labels).
- The datasets are merged using the `id` column.

## Project Structure
```
|-- datasets/                     # Folder containing dataset files
|-- notebooks/                     # Jupyter Notebook files (.ipynb)
|-- scripts/                       # Python scripts for model training
|-- README.md                      # Project documentation
|-- requirements.txt               # Required dependencies
```

## Installation & Setup
### 1️⃣ Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook
```bash
jupyter notebook
```

## Step-by-Step Workflow
### 1️⃣ Load Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```

### 2️⃣ Load and Merge Dataset
```python
train_data = pd.read_csv("datasets/train_data.csv")
churn_data = pd.read_csv("datasets/churn_data.csv")
merged_data = pd.merge(train_data, churn_data, on="id")
```

### 3️⃣ Data Preprocessing
```python
# Handle missing values
merged_data.fillna(merged_data.median(), inplace=True)

# Convert categorical variables to numerical
merged_data = pd.get_dummies(merged_data, drop_first=True)
```

### 4️⃣ Train-Test Split
```python
X = merged_data.drop(columns=["churn"])
y = merged_data["churn"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5️⃣ Model Training
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 6️⃣ Model Evaluation
```python
# Predictions
y_pred = model.predict(X_val)

# Performance Metrics
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
```

### 7️⃣ Confusion Matrix Visualization
```python
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Results & Insights
- The model successfully predicts customer churn with **high accuracy**.
- Further improvements can be made by **tuning hyperparameters** and **feature engineering**.

## Next Steps
- Optimize hyperparameters using `GridSearchCV`.
- Test other machine learning models such as **XGBoost or Logistic Regression**.

## Author
Prasiddha Pradhan

---
Feel free to contribute, improve, or suggest modifications!

