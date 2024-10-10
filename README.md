# 📊 Customer Churn Prediction Project
## 📫 Contact
If you have any questions or want to collaborate, feel free to reach out! LinkedIn: https://www.linkedin.com/in/ahmad-zaenal-hayat/ Email: a.zaenalhayat@gmail.com

## 📊 Project Overview
DQLab Telco is a telecommunications company with a global presence, dedicated to providing excellent customer experiences since its establishment in 2019. 
Despite its relatively short operational history, DQLab Telco has faced challenges with customer retention, as many customers have switched their subscriptions to rival companies. 
To address this issue, management aims to leverage machine learning techniques to reduce customer churn and improve overall satisfaction.

After completing the data cleansing process, the next step is to build an effective predictive model for forecasting customer churn.

You Access this dataset from kaggle https://www.kaggle.com/datasets/samran98/customer-churn-telco-final/data

## 🛠️ Tools & Libraries
For this project, the following tools and libraries were utilized:

Python: The primary programming language for data manipulation and analysis.
Pandas: A powerful library for data manipulation and analysis, providing data structures like DataFrames to manage and clean the dataset.
NumPy: A fundamental package for numerical computations in Python, useful for handling arrays and performing mathematical operations.
Scikit-learn: A key library for implementing machine learning algorithms, including Logistic Regression, data preprocessing, and model evaluation.
Statsmodels: Useful for statistical modeling, providing tools for estimating and testing the logistic regression model and its statistics.
Seaborn: A statistical data visualization library built on Matplotlib that helps create informative and attractive visualizations.
Matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.
Jupyter Notebook: The development environment used for writing and executing Python code, making it easier to visualize data and share insights.

## 🛠️ Tasks & Steps
To effectively predict customer churn, the following steps will be undertaken:

### 1. Exploratory Data Analysis (EDA):

Analyze the dataset to identify trends, patterns, and relationships between features and the churn variable.
Visualize the distribution of key attributes to understand customer demographics and behaviors.

### 2. Data Preprocessing:

Clean the dataset by handling missing values, converting categorical variables into numerical formats, and scaling features if necessary.
Split the data into training and testing sets for model evaluation.

### 3. Machine Learning Modeling:

Implement various machine learning algorithms (e.g., Logistic Regression, Decision Trees, Random Forests) to predict customer churn.
Fine-tune model parameters using cross-validation to enhance performance.

### 4. Selecting the Ideal Model:

Evaluate model performance using appropriate metrics (e.g., accuracy, precision, recall, F1 score).
Select the model that provides the best balance between accuracy and interpretability for deployment.

## 📈 Key Analysis: Logistic Regression & Customer Churn Prediction
The analysis focuses on building a logistic regression model to predict customer churn:

Data Preprocessing: Clean and preprocess the dataset, handling missing values and categorical variables.
Feature Selection: Identify and select relevant features that impact customer churn.
Model Training: Train a logistic regression model using the training dataset.
Model Evaluation: Evaluate model performance using metrics such as accuracy, precision, recall, and F1 score.

## 🔍 Key Findings from Exploratory Data Analysis (EDA)
Based on the results and analysis above, we can conclude:

In stage C.1, we can see that the overall distribution of customers indicates that they do not churn, with a churn rate of 26% and a no-churn rate of 74%.

In stage C.2, we observe that for Monthly Charges, there is a tendency that as the monthly fee decreases, the likelihood of churn also decreases. For Total Charges, there is no apparent trend regarding churn among customers. Additionally, for tenure, there is a tendency that the longer a customer has been subscribed, the lower the likelihood of churn.

In stage C.3, we find that there are no significant differences in churn based on gender and phone service factors. However, there is a tendency that those who churn are individuals who do not have a partner (Partner: No), are classified as senior citizens (SeniorCitizen: Yes), have streaming TV services (StreamingTV: Yes), have internet service (InternetService: Yes), and utilize paperless billing (PaperlessBilling: Yes).



## 🗂 Dataset Description
The dataset used for this analysis contains customer subscription information from June 2020. 
It includes various attributes related to customer demographics, subscription details, and churn status. Key features in the dataset include:

customerID: Unique identifier for each customer

gender: Gender of the customer (Male/Female)

SeniorCitizen: Indicates whether the customer is a senior citizen (1 = Yes, 0 = No)

Partner: Indicates whether the customer has a partner (Yes/No)

Dependents: Indicates whether the customer has dependents (Yes/No)

tenure: Number of months the customer has been with the company

PhoneService: Indicates if the customer has a phone service (Yes/No)

MultipleLines: Indicates if the customer has multiple lines (Yes/No)

InternetService: Type of internet service (DSL/Fiber optic/No)

OnlineSecurity: Indicates if the customer has online security (Yes/No)

OnlineBackup: Indicates if the customer has online backup (Yes/No)

DeviceProtection: Indicates if the customer has device protection (Yes/No)

TechSupport: Indicates if the customer has tech support (Yes/No)

StreamingTV: Indicates if the customer has streaming TV service (Yes/No)

StreamingMovies: Indicates if the customer has streaming movies service (Yes/No)

Contract: Type of contract (Month-to-month/One year/Two year)

PaperlessBilling: Indicates if the customer uses paperless billing (Yes/No)

PaymentMethod: Payment method used (e.g., Credit card, Bank transfer)

MonthlyCharges: Amount charged to the customer each month

TotalCharges: Total amount charged to the customer

Churn: Indicates whether the customer has churned (Yes/No)
