# Bank Term Deposit Subscription Prediction
### Machine Learning Project with Real-World Business Impact & Streamlit Deployment

This project builds a machine learning solution to help banks predict whether a customer is likely to subscribe to a term deposit.
Rather than calling thousands of customers blindly, this model enables marketing teams to focus their efforts on customers with higher conversion potential, resulting in improved campaign efficiency and reduced operational cost.

The goal was not only to train a model with high accuracy, but to design a solution that reflects how predictive analytics is used in real business decision-making from preprocessing and model evaluation to threshold tuning and deployment.

---

## Problem Context

Banks run telemarketing campaigns to promote long-term deposit products. These campaigns are costly, repetitive, and have historically low success rates (≈11% positive response).

The challenge:

➡ How can we identify which customers are most likely to subscribe before calling them?

This project answers that by learning from historical campaign data and predicting subscription outcomes ahead of time.

The project includes:

- End-to-end machine learning pipeline (preprocessing → feature engineering → training → evaluation)

- Smart handling of class imbalance

- Custom decision threshold tuned for better recall (the business goal)

- Feature importance insights explain why predictions are made

- Fully deployed as a Streamlit web application

- Clean, reusable, production-style code structure

---

## Machine Learning Pipeline

### 1. Data Understanding & Cleaning

- Loaded the Bank Marketing dataset containing telemarketing campaign history.

- Checked distributions, datatypes, and class imbalance.

- Removed the duration column because it leaks information (it is only known after a call is completed and cannot be used for prediction).

### 2. Feature Engineering

- Created a meaningful binary feature:

  - was_contacted_before → derived from pdays, indicating if the customer was ever contacted in a previous campaign.

- This helped the model understand customer familiarity with the bank’s offers — an important predictor.

### 3️. Preprocessing

Used a ColumnTransformer to ensure clean and consistent input for the model:


| Feature Type | Transformation                            |
| ------------ | ----------------------------------------- |
| Categorical  | One-Hot Encoding + Missing value handling |
| Numerical    | Median imputation + scaling               |


This step converts human-readable information into machine-understandable form without losing meaning.

### 4️. Handling Imbalance

Since only ~11% of customers subscribed previously, the model could become biased toward predicting “no”.

- To counter this, class weighting was applied during model training.

- The objective was to make the model pay fair attention to the minority (YES) class.

### 5️. Model Selection & Training

A Random Forest Classifier was selected because:

- It performs well with mixed data types

- It naturally handles nonlinearity

- It provides feature importance for interpretation

The model was trained inside a full pipeline to ensure preprocessing, feature encoding, and modeling occur consistently.

### 6️. Model Evaluation

Evaluated using beyond just accuracy:

- Classification Report

- ROC–AUC

- Precision–Recall AUC

- Confusion Matrix

- Metrics evaluated before and after threshold adjustment

This ensured the model was not only mathematically correct but practically valuable.

### 7️. Decision Threshold Tuning

The default probability threshold (0.50) resulted in low recall for the positive class.
After testing multiple thresholds, a lower threshold was selected to capture more potential subscribers — aligning the model with the actual business objective.

### 8️. Deployment with Streamlit

The final model was deployed as a user-friendly web application, allowing real-time prediction based on customer details.
Marketing teams can now use the app to make informed decisions, quickly assess customers, and improve campaign efficiency.

---

## Evaluation Summary

The model performed well across key metrics:


| Metric               | Result                 |
| -------------------- | ---------------------- |
| Accuracy             | ~88%                   |
| ROC-AUC              | ~0.80                  |
| Precision–Recall AUC | ~0.48                  |
| Recall After Tuning  | Improved significantly |

The threshold was intentionally adjusted because in this business problem, missing a potential subscriber costs more than calling one extra customer.

## Installation 

Clone the repository:

```bash
git clone https://github.com/KasinathMS/Bank-Term-Deposit-Subscription-Prediction.git
cd Bank-Term-Deposit-Subscription-Prediction
```
Install the required dependencies:

```bash
pip install pandas numpy scikit-learn streamlit matplotlib joblib
```
## Run the Application

To launch the Streamlit app:

```bash
streamlit run app.py
```

## Technologies Used

- Python
- Scikit-Learn
- Pandas, NumPy
- Streamlit
- Matplotlib
- Joblib



