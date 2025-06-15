# 🧪 A Multilingual System for Early-Stage Diabetes Risk Prediction Using Machine Learning Approaches

📌 **Project Overview**

This project develops a multilingual system to predict the risk of early-stage diabetes using machine learning. By offering support for multiple languages, it aims to improve accessibility for non-English speakers and facilitate early diagnosis and intervention. The process involves thorough data preparation, exploratory data analysis, feature engineering, model selection, refinement, and deployment as a user-friendly web application.

🎯 **Objectives**

* ✅ Predict the risk of diabetes based on individual health data.
* 🌐 Provide a user-friendly interface with multilingual support (initially English and Amharic).
* 🌱 Support early intervention and promote health equity across diverse populations.
* 🔬 Implement and evaluate various machine learning models for early-stage diabetes prediction.
* ⚙️ Deploy the best-performing model within a multilingual web application using Flask.

🌍 **SDG Relevance**

* **SDG 3 – Good Health & Well-Being:** This project contributes to early detection and management of diabetes, promoting better health outcomes.
* **SDG 10 – Reduced Inequalities:** By offering a multilingual tool, it enhances healthcare accessibility for individuals from various linguistic backgrounds.

🔍 **Dataset**

* **Name:** Early Stage Diabetes Risk Prediction Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)
* **Size:** 520 records
* **Features:** 17 attributes including Age, Gender, and 15 symptoms (Polyuria, Polydipsia, Sudden weight loss, etc.).
* **Target Variable:** Binary (Positive for early diabetes symptoms, Negative for no symptoms).

**🛠️ Data Preparation & Feature Engineering**

1.  **Overview:** Ensuring raw data is clean, formatted, and structured for effective learning.
2.  **Data Collection:** Loaded the dataset using Pandas.
3.  **Data Cleaning:**
    * No missing values were found.
    * No extreme outliers in the 'Age' variable.
    * Categorical features ('Yes'/'No', 'Male'/'Female', 'Positive'/'Negative') were encoded to binary (0/1).
    * Categorical string values were standardized to lowercase.
4.  **Exploratory Data Analysis (EDA):**
    * **Age Distribution:** Bimodal distribution with peaks around 40 and 60 years.
    * **Gender Distribution:** Mild gender imbalance with more male participants.
    * **Class Distribution:** Slight imbalance with the 'positive' class being slightly more frequent.
    * **Feature Correlation:** 'Polyuria' and 'Polydipsia' showed strong positive correlation with the target class.
5.  **Feature Engineering:**
    * Categorical variables and the target variable were binary encoded.
    * Chi-squared test identified 'Polydipsia', 'Polyuria', and 'sudden weight loss' as top-scoring features.
    * Feature importance was visualized after model training.
6.  **Data Transformation:**
    * 'Age' was scaled using `StandardScaler`. Binary features were already effectively scaled (0 or 1).

**🤖 Model Exploration**

1.  **Model Selection:** Ten diverse classification algorithms were explored: Logistic Regression, LDA, KNN, Decision Tree, SVM, Random Forest, AdaBoost, XGBoost, Naive Bayes, and MLP.
2.  **Model Training:** Models were trained using Stratified K-Fold cross-validation. The Random Forest model underwent hyperparameter tuning using `RandomizedSearchCV`.
3.  **Model Evaluation:** Performance was evaluated using Accuracy, Precision, Recall, F1-Score, and ROC AUC. Visualizations included Confusion Matrices and ROC Curves.

**⚙️ Model Refinement**

1.  **Overview:** Optimizing model performance through techniques like hyperparameter tuning.
2.  **Model Evaluation:** Initial evaluation highlighted the Random Forest Classifier's strong performance.
3.  **Refinement Techniques:** Primarily focused on hyperparameter tuning of the Random Forest using `RandomizedSearchCV` and addressing class imbalance with `RandomOverSampler`.
4.  **Hyperparameter Tuning:** Best parameters found for Random Forest were `{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 16, 'criterion': 'gini'}`, achieving a ROC AUC of approximately 0.997.
5.  **Cross-Validation:** Stratified K-Fold (10 splits initially, 5 within `RandomizedSearchCV`).
6.  **Feature Selection:** While explored, all features were used in the final tuned model.

**🧪 Test Submission**

1.  **Overview:** Evaluating the final tuned model on a held-out test dataset.
2.  **Data Preparation for Testing:** Test data was split (20%), scaled using the fitted `StandardScaler`, and class imbalance was not addressed in the test set.
3.  **Model Application:** The tuned Random Forest Classifier was used to make predictions.
4.  **Test Metrics:** Achieved approximately 99.04% accuracy and an AUC of 1.00 on the test set.
5.  **Model Deployment:** The tuned model was serialized using `pickle`. A multilingual web application using Flask was developed with language selection (English and Amharic).
6.  **Code Implementation:** Relevant code snippets for data processing, model training, and evaluation are available in the project files.

**🚀 Deployment**

The trained model is deployed as a multilingual web application using Flask, allowing users to input their information and receive a diabetes risk prediction in their chosen language (English or Amharic).

**📚 References**

* [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)
* [Imbalanced-learn: A Python package to tackle the class imbalance problem](https://imbalanced-learn.org/stable/)
* [UCI Machine Learning Repository: Early Stage Diabetes Risk Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Pickle (Python documentation)](https://docs.python.org/3/library/pickle.html)
* [Flask](https://flask.palletsprojects.com/)
