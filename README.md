# Road Accident Severity Prediction

A machine learning project for predicting road accident severity across selected Indian National Highways using historical accident data.

## Overview

Road accidents are influenced by several factors, including road conditions, weather, vehicle types, accident characteristics, and time. Understanding these factors can help identify patterns associated with different accident severity levels.

This project develops and evaluates multiple machine learning classification models to predict accident severity. It also includes an interactive Streamlit application that allows users to enter accident-related details and obtain model-generated predictions.

## Objectives

- Analyse historical road accident data from selected Indian National Highways.
- Explore accident patterns through data analysis and visualisation.
- Perform data preprocessing and feature selection.
- Develop and compare multiple machine learning classification models.
- Evaluate model performance using class-sensitive evaluation metrics.
- Build an interactive application for accident severity prediction.

## Dataset

The project uses the **Accident Severity Prediction Modelling for Indian Highways** dataset.

| Attribute | Details |
|---|---|
| Source | Zenodo |
| Study areas | Pune–Solapur NH-9 and Bengal NH-2 |
| Records in the project copy | 3,253 |
| Original features | 35 |
| Target variable | Accident Severity |

**Dataset:** [Zenodo – Accident Severity Prediction Modelling for Indian Highways](https://zenodo.org/records/7773156)

The dataset was obtained from previously published research. This project independently explores the data and evaluates different machine learning approaches.

## Accident Severity Categories

The target variable consists of four classes:

| Class | Severity |
|---|---|
| 1 | Fatal |
| 2 | Grievous Injury |
| 3 | Minor Injury |
| 4 | No Injury |

## Methodology

The project follows these major stages:

1. **Data Collection:** Obtaining the accident dataset from Zenodo.
2. **Data Preprocessing:** Handling missing values, correcting identified data inconsistencies, and preparing categorical and numerical features.
3. **Exploratory Data Analysis:** Analysing accident distributions and relationships between accident characteristics and severity.
4. **Feature Selection:** Selecting relevant accident-related features while excluding variables that could introduce target leakage.
5. **Model Development:** Training and comparing multiple classification algorithms.
6. **Model Evaluation:** Evaluating performance using classification metrics, confusion matrices, and cross-validation.
7. **Application Development:** Integrating trained models into an interactive Streamlit application.

## Machine Learning Models

The following algorithms were implemented:

- Decision Tree
- Random Forest
- Gradient Boosting
- Improved Random Forest
- Fatal-focused Random Forest (additional experiment)

The Improved Random Forest uses class-weight balancing and adjusted model parameters to improve performance across severity categories.

## Model Performance

The four standard models were evaluated on the same test dataset.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 36.41% | 30.80% | 31.12% | 30.88% |
| Random Forest | 43.78% | 36.76% | 36.82% | 36.78% |
| Gradient Boosting | **47.16%** | **43.14%** | 35.92% | 36.61% |
| Improved Random Forest | 44.09% | 38.45% | **37.94%** | **38.15%** |

### Selected Model: Improved Random Forest

The Improved Random Forest achieved the highest Macro F1-score among the four standard models.

- **Test Macro F1-score:** 0.3815
- **5-Fold Cross-Validation Macro F1-score:** 0.3827 ± 0.0115
- **5-Fold Cross-Validation Accuracy:** 0.4458 ± 0.0099

Although Gradient Boosting achieved the highest accuracy, the Improved Random Forest was selected because it provided better overall balance across the four severity classes, as reflected by its Macro F1-score.

### Additional Experiment: Fatal-focused Random Forest

An additional experiment explored class weighting to improve the detection of fatal accidents.

- **Fatal class recall:** 35%
- **Macro F1-score:** 0.3649

This experiment demonstrated a trade-off between detecting fatal accidents and maintaining balanced performance across all severity categories.

## Evaluation Metrics

The following metrics were used:

- **Accuracy:** Overall proportion of correctly classified accidents.
- **Macro Precision:** Average precision across all severity classes.
- **Macro Recall:** Average recall across all severity classes.
- **Macro F1-score:** Balanced measure of precision and recall across severity classes.
- **Confusion Matrix:** Examines correct and incorrect predictions for each class.
- **Stratified Cross-Validation:** Evaluates model consistency across multiple training and validation splits.

Macro-averaged metrics are particularly important because the dataset contains an unequal distribution of accident severity classes.

## Interactive Application

The project includes a Streamlit-based application for accident severity prediction.

**Application features:**

- Selection between four trained machine learning models.
- Input of accident-related characteristics.
- Prediction of one of four accident severity categories.
- Visualisation of model-estimated class probabilities.
- Simple interface for interactive predictions.

The application is intended for educational and research purposes. Predictions should not be treated as definitive assessments of real-world accident outcomes.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
Accident-Severity-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
│
├── data/
│   └── raw/
│   
│
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── improved_random_forest.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── outputs/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Accident-Severity-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Limitations

- The dataset represents selected highway sections and may not generalise to all Indian roads.
- The dataset has an imbalanced distribution across accident severity categories.
- Model performance depends on the quality and availability of recorded accident information.
- Predictions are estimates based on historical patterns and do not establish causal relationships.
- The application is a research prototype, not a real-world accident prevention or emergency response system.

## Research Reference

**Original dataset:**

Khanum, H., Garg, A., & Faheem, M. I. Data for Accident Severity Prediction Modelling for Indian Highways Case Study.

[Zenodo Dataset](https://doi.org/10.5281/zenodo.7773156)

---

*Developed as a machine learning project exploring road accident severity prediction using Indian highway accident data.*