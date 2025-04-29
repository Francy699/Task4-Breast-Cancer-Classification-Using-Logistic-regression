
# Breast Cancer Classification using Logistic Regression

This project demonstrates the classification of breast cancer cases into benign and malignant categories using Logistic Regression. The dataset used contains various features related to cell measurements, and the target variable indicates whether a tumor is benign or malignant.

## Concept of Binary Classification

Binary classification is a type of classification problem where the goal is to classify data into one of two possible classes or categories. In this project, the two classes are:
- **Malignant (M)**: Malignant tumors are cancerous and harmful.
- **Benign (B)**: Benign tumors are non-cancerous and not harmful.

The task is to classify each tumor as either malignant or benign based on a set of features (cell measurements).

## Concept of Logistic Regression

Logistic Regression is a statistical method used for binary classification tasks. Unlike linear regression, which outputs continuous values, logistic regression predicts the probability that an observation belongs to one of the two classes. The output is a value between 0 and 1, which is interpreted as the probability of the observation being in the positive class (malignant tumor in this case).

The formula for logistic regression is:
```
P(Y=1|X) = 1 / (1 + e^(-Z))
```
Where:
- **P(Y=1|X)** is the probability of the positive class (malignant).
- **Z** is the linear combination of the input features (X) and model coefficients.
- **e** is Euler's number (approximately 2.718).

Logistic regression uses a sigmoid function (also called the logistic function) to model the probability of a binary outcome.

---

## 📦 Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
```

## 📥 Load Dataset

We load the dataset, clean it by dropping irrelevant columns, and encode the target variable (diagnosis).

```python
df = pd.read_csv("data.csv")  # Update path if needed
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)  # Drop irrelevant columns
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode target
```

## 🔍 Features and Target

We define the features (X) and the target (y).

```python
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
```

## 🔀 Train-Test Split

The dataset is split into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 📏 Scale Features

The features are scaled using `StandardScaler` to standardize the data for better model performance.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 🤖 Train Logistic Regression Model

The Logistic Regression model is trained using the scaled training data.

```python
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)
```

## 🔮 Make Predictions

Predictions and probability estimates are made on the test data.

```python
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

## 📊 Confusion Matrix

A confusion matrix is displayed to evaluate the model's performance on the test set.

```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## 📄 Classification Report

The classification report, which includes precision, recall, f1-score, and support, is printed to evaluate the model.

```python
print("Classification Report:
")
print(classification_report(y_test, y_pred))
```

## 📈 ROC Curve

The ROC curve and AUC score are plotted to evaluate the model's performance in distinguishing between the two classes.

```python
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
```

## 📉 Precision-Recall Curve

The precision-recall curve is plotted to show the trade-off between precision and recall at different thresholds.

```python
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="purple")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()
```

## 📌 Feature Importance (Coefficients)

The coefficients of the logistic regression model are plotted to show the importance of each feature.

```python
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance_df)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.tight_layout()
plt.show()
```

## 🧠 Print AUC Score

Finally, the ROC-AUC score is printed as a measure of the model's performance.

```python
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
```

---

### Requirements
To run the code, make sure to have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Dataset

The dataset used in this project should be a CSV file named `data.csv`. Ensure it has the following structure:
- The columns `id` and `Unnamed: 32` should be removed.
- The target column (`diagnosis`) should have values 'M' (Malignant) and 'B' (Benign).

