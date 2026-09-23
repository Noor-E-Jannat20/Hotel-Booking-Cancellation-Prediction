# CSE422 Hotel Booking Cancellation Prediction

## Overview
A machine learning project that predicts whether a hotel booking will be canceled using the **Hotel Booking Demand** dataset.

**Dataset:** 119,390 records, 32 features  
**Target:** `is_canceled` — 0 = Not Canceled, 1 = Canceled

## Models
- KNN
- Decision Tree
- Logistic Regression
- Naive Bayes
- Neural Network (MLP)
- K-Means Clustering

## Workflow
```text
Data Loading → EDA → Preprocessing → Train/Test Split
→ Model Training → Evaluation → Comparison
```

## Preprocessing
- Removed `reservation_status` and `reservation_status_date` to prevent target leakage.
- Dropped `company` because of its high percentage of missing values.
- Filled remaining missing values.
- Applied Label Encoding to categorical features.
- Applied StandardScaler for KNN, Logistic Regression, and MLP.
- Used an 80/20 stratified train/test split.

## Evaluation
Models are compared using Accuracy, Precision, Recall, ROC-AUC, Confusion Matrix, and ROC Curves.

| Model | Accuracy | Precision | Recall | AUC |
|---|---:|---:|---:|---:|
| KNN | 0.8330 | 0.7912 | 0.7460 | 0.9046 |
| Decision Tree | 0.8404 | 0.8248 | 0.7228 | 0.9185 |
| Logistic Regression | 0.7933 | 0.8045 | 0.5838 | 0.8637 |
| Naive Bayes | 0.5709 | 0.4597 | 0.9034 | 0.8013 |
| Neural Network (MLP) | 0.8559 | 0.8201 | 0.7826 | 0.9345 |

## Key Insights
Important features identified in the notebook include:
`lead_time`, `deposit_type`, `previous_cancellations`, and `total_of_special_requests`.

## Requirements
Designed for **Google Colab**.

Main libraries:
```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

## How to Run
1. Open `cse422_hotel_project_Grp01_colab.ipynb` in Google Colab.
2. Make sure `hotel_bookings.csv` is available in Google Drive.
3. Update the dataset path if necessary.
4. Run the notebook cells from top to bottom.

## Project Files
```text
cse422_hotel_project_Grp01_colab.ipynb
hotel_bookings.csv
README.md
```
