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

### Jupyter Notebooks
The `notebooks/` folder contains step-by-step development and exploration:

1. `01-data-exploration.ipynb` – Explore the dataset and visualize anomalies  
2. `02-preprocessing.ipynb` – Perform ETL and cleaning steps  
3. `03-ml-training.ipynb` – Train and evaluate the anomaly detection model  

### FastAPI
Run the API locally:

    python src/api.py

Example request:

    POST /predict
    Content-Type: application/json

    [
        {
            "feature1": 10,
            "feature2": "A",
            "feature3": 3.14
        }
    ]

Example response:

    [
        {
            "anomaly": true,
            "score": -0.65
        }
    ]

---

## Project Structure

    ai-etl-anomaly-detection/
        data/
        notebooks/
        src/
            data_loader.py
            preprocessing.py
            feature_engineering.py
            model.py
            evaluate.py
            api.py
        models/
        tests/
        Pipfile
        Pipfile.lock
        requirements.txt (optional)
        README.md
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
