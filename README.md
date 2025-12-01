# Bank Term Deposit Subscription Prediction
### An end-to-end Machine Learning project with Streamlit deployment

This project predicts whether a bank customer is likely to subscribe to a **term deposit** based on demographic, financial, contact,and economic indicators. It uses the popular **Bank Marketing** dataset and demonstrates the full ML pipeline: preprocessing, feature engineering, model training, imbalance handling, evaluation, and deployment using Streamlit.

---

## Project Overview

Banks often run marketing campaigns to promote term deposits. However, contacting every customer is expensive and inefficient.  
This project builds a machine learning model that analyzes past campaign data and **predicts which customers are most likely to subscribe**, helping marketing teams prioritize the right leads.

The project includes:

- Data preprocessing & cleaning  
- Feature engineering  
- Handling class imbalance (SMOTE)  
- Random Forest model training  
- ROC-based threshold tuning  
- Deployment as an interactive **Streamlit web app**

---

## Machine Learning Pipeline

### **1. Data Preprocessing**
- Removed leakage column: `duration`
- Added engineered feature:
  - `was_contacted_before` (derived from `pdays`)
- Categorical features encoded using **OneHotEncoder**
- Numerical features passed through untouched or scaled as needed

---

### **2. Handling Class Imbalance**
The dataset is highly imbalanced (`yes` ≈ 11%).  
To address this, **SMOTE** was applied to the transformed features after preprocessing.

---

### **3. Model Training**
A **Random Forest Classifier** was chosen due to its:

- Strong performance with mixed data types  
- Ability to capture nonlinear relationships  
- Robustness to noise and overfitting  

The model was trained on the SMOTE-balanced dataset.

---

### **4. Model Evaluation**

Key performance results:

- **Accuracy:** ~94%  
- **AUC-ROC:** ~0.97  
- **Balanced precision-recall** after threshold tuning

Although the AUC score was excellent, the default threshold of 0.5 produced very few positive predictions due to class imbalance.  
Thus, a more practical threshold of **0.4** was selected based on ROC curve analysis.

---

## Decision Threshold

During evaluation, many true "yes" customers received predicted probabilities in the **0.35–0.49** range.  
This indicates that:

- The model correctly identifies positive cases  
- But does not push them above 0.5

With threshold = 0.4:

- The model becomes more practical for marketing campaigns  
- Recall of the positive class improves significantly  
- Predictions become more balanced and actionable  

This threshold is implemented in streamlit.

---
## Technology Stack

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Streamlit
- Matplotlib 



