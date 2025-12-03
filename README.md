# AI-Driven ETL Anomaly Detection

[![Python](https://img.shields.io/badge/python-3.14-blue)](https://www.python.org/downloads/latest/python3/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99-green)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange)](https://scikit-learn.org/)


## Project Overview
This project demonstrates an **AI-powered ETL pipeline** that automatically detects anomalies and data quality issues in structured datasets. It integrates **data ingestion, preprocessing, machine learning-based anomaly detection, and a FastAPI deployment** to provide actionable insights.

The repository is designed to showcase skills in:

- Python-based ETL pipelines  
- Machine Learning (anomaly detection using Scikit-Learn)  
- Backend deployment with FastAPI  
- Data quality monitoring and reporting  
- Clean, professional project structure for enterprise-level applications

---

## Features

- Ingest data from CSV files or databases  
- Perform data cleaning and preprocessing  
- Feature engineering for anomaly detection  
- Train and evaluate an ML model to detect anomalies  
- Generate reports highlighting data quality issues  
- Expose a FastAPI `/predict` endpoint for real-time anomaly scoring  

---

## Data

### Synthetic Transactions Dataset
- Path: `data/raw/synthetic_transactions.csv`
- Includes:
  - Normal transactions
  - Injected anomalies: large amounts, negative/zero values, category deviations
- Fully included in the repo for exploration and modeling

### Kaggle Credit Card Fraud Dataset (Not Included)
- Original dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Note:** `creditcard.csv` is too large for GitHub, so it is **not included** in this repo.
- To use the Kaggle dataset locally:
  1. Sign in to Kaggle and download `creditcard.csv`.
  2. Place it in the folder: `data/raw/creditcard.csv`
  3. The Day 3 notebook will automatically load it from this path.

```python
# Example: loading Kaggle data
import pandas as pd

df_kaggle = pd.read_csv('data/raw/creditcard.csv')
```

---

## Installation

This project uses **Pipenv** for dependency management:

    pipenv install --dev
    pipenv shell

Alternatively, if you prefer `pip`:

    pip install -r requirements.txt

---

## Usage

### Notebooks Overview
This project includes a set of structured Jupyter notebooks that walk through the full lifecycle of the anomaly detection pipeline:

#### 01-data-exploration.ipynb
Initial EDA, anomaly visualization, data distributions, missing values, and exploratory insights.

#### 02-preprocessing.ipynb
ETL pipeline construction, cleaning, scaling, handling skewed features, and feature engineering.

#### 03-ml-training.ipynb
Model training (Isolation Forest or others), tuning, evaluation metrics, ROC/AUC, and result interpretation.

A detailed explanation for each notebook is provided inside the notebooks/README.md to help reviewers understand design decisions and methodology.  

## API Documentation

This project includes a production-ready FastAPI microservice that exposes the unified anomaly detection pipeline for real-time and batch inference.

### Base URL

```text
http://localhost:8000
```

### Endpoints
**GET /health**
Simple heartbeat.

**Response**
```json
{ "status": "ok", "message": "Anomaly Detection API is running." }
```

**Get /metadata**
Returns model, scaler, and preprocessor metadata.

**POST /predict**
Perform real-time anomaly detection on one transaction.

**Request**
```json
{
  "timestamp": "2025-01-01T11:22:00",
  "customer_id": 101,
  "Amount": 129.55,
  "category": "grocery",
  "status": 0
}
```

**Response**
```json
{
  "prediction": 0,
  "anomaly_score": -0.21
}
```

**POST /predict_batch**
Perform anomaly detection on multiple transactions.

**Request**
```json
{
  "records": [
    { "timestamp": "...", "Amount": 129.55, "category": "grocery" },
    { "timestamp": "...", "Amount": 980.25, "category": "tech" }
  ]
}
```

**Response**
```json
{
  "count": 2,
  "predictions": [0, 1],
  "anomaly_scores": [-0.21, 0.88]
}
```

### Running The API Locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Running Docker:

```bash
docker build -t anomaly-api .
docker run -p 8000:8000 anomaly-api
```

## Project Structure

    ai-etl-anomaly-detection/
        data/
            raw/
            processed/
            results/
        models/
        notebooks/
        src/
            BaseCLasses/
                base_preprocessor.py
            Preprocessors/
                kaggle_preprocessor.py
                synthetic_preprocessor.py
                unified_preprocessor.py
            data_loader.py
            preprocessing.py
            feature_engineering.py
            model.py
            evaluate.py
            api.py
        tests/
        Pipfile
        Pipfile.lock
        requirements.txt (optional)
        README.md (this file)
        .gitignore

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Next Steps / Enhancements

- Add automated ETL orchestration with **Airflow**  
- Implement **real-time anomaly monitoring dashboards**  
- Include additional ML models (e.g., Autoencoders) for advanced anomaly detection  
- Deploy API to **cloud services** (AWS, GCP, Azure)  

---

## Contact / Author

D Fashimpaur  
[LinkedIn](https://www.linkedin.com/in/dfashimpaur)
