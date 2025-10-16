# 🏠 House Price Prediction System

## 📖 Overview  
The **House Price Prediction System** is a machine learning project designed to predict the price of a house based on various features such as area, number of rooms, location, and other property attributes.  

This project implements and compares multiple regression algorithms — **Linear Regression**, **Lasso Regression**, **Ridge Regression**, and **ElasticNet Regression** — to determine the most accurate model.  

The pipeline includes **Exploratory Data Analysis (EDA)**, **feature engineering**, **statistical assumption validation**, and **model evaluation** using metrics such as **R²** and **RMSE**. The best model is **serialized using Pickle** and **deployed via Flask API** on **AWS EC2** for real-time predictions.

---

## ✨ Key Features
- 🔍 **Exploratory Data Analysis (EDA)** to identify trends and outliers  
- ⚙️ **Feature Engineering** and **data preprocessing** for model optimization  
- 📊 **Multiple Regression Techniques:** Linear, Lasso, Ridge, ElasticNet  
- 📈 **Model Evaluation** using Cross-Validation, R², and RMSE  
- 🧠 **Model Serialization** using Pickle for deployment  
- 🌐 **Flask API Integration** for real-time predictions  
- ☁️ **Deployment on AWS EC2** for scalable, production-ready performance  

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python |
| Machine Learning | scikit-learn, NumPy, pandas |
| Data Visualization | Matplotlib, Seaborn |
| Backend Framework | Flask |
| Deployment | AWS EC2 |
| Model Serialization | Pickle |
| Development | Jupyter Notebook / VS Code |

---

## 📂 Project Structure
House-Price-Prediction/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── notebooks/
│ └── EDA_and_Model_Training.ipynb
│
├── model/
│ └── best_model.pkl
│
├── app/
│ ├── app.py # Flask API for prediction
│ ├── templates/
│ │ └── index.html # Frontend form (if included)
│ └── static/
│ └── style.css # Optional CSS
│
├── requirements.txt
├── README.md
