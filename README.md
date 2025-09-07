# 📊 Credit Risk Prediction – ML + Streamlit  

A machine learning project that predicts **loan default risk** using multiple models (*Logistic Regression, Random Forest, and XGBoost*) and provides a **Streamlit web app** for real-time predictions.  

## 🚀 Live Demo
👉 [Click here to try the app](https://credit-risk-prediction-application.streamlit.app/)

Dataset: [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  

---

## 🚀 Tech Stack  
- **Machine Learning:** scikit-learn, XGBoost  
- **Frontend:** Streamlit (interactive UI)  
- **Backend:** Python  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Model Persistence:** Joblib  

---

## 🔑 Core Features  

### 📊 Model Training  
- Preprocessing with scaling and encoding  
- Trains **Logistic Regression, Random Forest, XGBoost**  
- Compares models using *Accuracy, F1 Score, and ROC-AUC*  
- Saves best-performing model and preprocessing objects  

### 🌐 Streamlit Web App  
- User-friendly interface for entering loan details  
- Encodes categorical variables using saved encoders  
- Scales numeric inputs with the trained scaler  
- Predicts:  
  - ✅ **Loan Default:** Yes / No  
  - 📈 **Probability of Default**  

### 📈 Model Comparison  
- Evaluate performance of multiple ML models  
- Visualize ROC curves for comparison  

---

## 📂 Folder Structure  

```plaintext
CreditRisk/
├── app.py                   # Streamlit app for prediction
├── credit_risk_training.py  # Model training & evaluation
├── xgboost_model.pkl        # Trained XGBoost model
├── random_forest_model.pkl  # Trained Random Forest model
├── logistic_regression.pkl  # Trained Logistic Regression model
├── scaler.pkl               # StandardScaler for numeric features
├── encoders.pkl             # LabelEncoders for categorical features
├── feature_order.pkl        # Column order for consistent input
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

```

## 🌐 Project Setup

To set up and run this project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/credit-risk-prediction.git
   cd credit-risk-prediction
   ```

2. **Create Virtual Environment (Optional but Recommended):**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install Dependencies:**:
    ```bash
     pip install -r requirements.txt
     ```

4. **Run Model Training:**:
   ```bash
   python credit_risk_training.py
   ```
   **This will:** 
     - ✅ **Preprocess data** 
     - ✅ **Train Logistic Regression, Random Forest, and XGBoost models**
     - ✅ **Save trained models and preprocessing files**


5. **Run Streamlit App:**
   ```bash
   streamlit run app.py
   ```  
   **Accepts the app at:** 
      - **Local URL: `http://localhost:8501`**  


  ## 📌 Future Enhancements

- ⚙️ **Hyperparameter Tuning:** Use `GridSearchCV` or `Optuna` to optimize model performance.  
- ✨ **Feature Engineering:** Create new features or transform existing ones to improve accuracy.  
- 🌐 **API Integration:** Build REST API endpoints for integration with web apps.  




