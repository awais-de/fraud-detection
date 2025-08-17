# 🕵️ Fraud Detection System

A machine learning pipeline built in **Python** for detecting fraudulent transactions.  
The system applies **data preprocessing, feature engineering, model training, and evaluation** using popular ML libraries like **Scikit-Learn, Pandas, Matplotlib, and XGBoost**.  

---

## 📂 Project Structure

fraud-detection/
│
├── data/ # Raw and processed datasets
│ ├── raw/ # Original datasets (e.g., Kaggle CSVs)
│ ├── processed/ # Cleaned & feature-engineered datasets
│ └── external/ # Synthetic or additional sources
│
├── notebooks/ # Jupyter notebooks for exploration
│ ├── 01_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_visualizations.ipynb
│
├── src/ # Core source code
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── evaluation.py
│ └── utils.py
│
├── models/ # Saved trained models
│ └── fraud_model.pkl
│
├── reports/ # Results and visualizations
│ ├── figures/
│ └── metrics.txt
│
├── scripts/ # Executable scripts
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── predict_new.py
│
├── requirements.txt # Python dependencies
├── config.yaml # Configurations (paths, hyperparameters, etc.)
├── README.md # Project overview
└── .gitignore # Ignore unnecessary files



---

## ⚡ Features

- Data preprocessing: cleaning, encoding, scaling  
- Feature engineering: user-based and transaction-based features  
- Class imbalance handling with **SMOTE / class weighting**  
- Models: Logistic Regression, Random Forest, XGBoost  
- Evaluation: Precision, Recall, F1-score, ROC-AUC  
- Visualization: Confusion matrix, ROC curve, fraud patterns  
- Model persistence with **Joblib**  
- Extensible pipeline for deployment  

---

## 📊 Dataset Options

You can use one of these datasets:  

1. **Credit Card Fraud Detection (Kaggle)**  
   - [Dataset Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
   - 284,807 transactions, 492 frauds (imbalanced dataset).  

2. **IEEE-CIS Fraud Detection (Kaggle)**  
   - [Dataset Link](https://www.kaggle.com/c/ieee-fraud-detection)  
   - Rich features with 1M+ transactions.  

3. **PaySim Mobile Money Transactions**  
   - [Dataset Link](https://www.kaggle.com/ntnu-testimon/paysim1)  
   - Simulated transactions with fraud labels.  

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```


### 2️⃣ Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


### 3️⃣ Install dependencies
pip install -r requirements.txt


### 4️⃣ Download dataset
Place raw datasets (CSV files) in data/raw/.

## ⚙️ Usage

### Train the model
python scripts/train_model.py

### Evaluate the model
python scripts/evaluate_model.py

### Predict new transactions
python scripts/predict_new.py --input data/processed/new_transactions.csv

## 👨‍💻 Author
Muhammad Awais (@awais-de)