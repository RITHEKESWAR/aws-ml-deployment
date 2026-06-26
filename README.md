# AWS MLOps Pipeline

End-to-end Machine Learning deployment on AWS using SageMaker, Docker & GitHub Actions.

![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazon-aws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

## Features
- Model training pipeline
- Containerized inference
- CI/CD with GitHub Actions
- Ready for AWS SageMaker / EC2 / Lambda

## Tech Stack
- AWS (SageMaker, S3, ECR)
- Scikit-learn / XGBoost
- Docker
- GitHub Actions

## Project Structure# aws-ml-deployment
Aspiring AI &amp; Cloud Engineer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Sample dataset (replace with real data)
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [10, 20, 30, 40, 50, 60],
    'target': [0, 0, 0, 1, 1, 1]
})

X = data[['feature1', 'feature2']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
print("Model saved to model/model.joblib")
