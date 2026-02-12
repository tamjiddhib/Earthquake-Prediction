# 🌍 Earthquake-Prediction Using Seismic Data  
Machine Learning based earthquake time-to-failure prediction using LANL seismic dataset.
### Machine Learning Based Seismic Time-to-Failure Estimation

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-green)
![Dataset](https://img.shields.io/badge/Dataset-LANL%20Earthquake-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

This project applies machine learning techniques to analyze high-frequency seismic acoustic signals and estimate the time remaining before an earthquake occurs.  

By extracting meaningful statistical features from time-series data, the model identifies hidden vibration patterns that appear before seismic failure.

---

## 📊 Dataset Information

- **Dataset:** LANL Earthquake Prediction  
- **Source:** Kaggle Competition  
- **Type:** Time-Series Seismic Data  
- **Target Variable:** Time to Failure  
- **Feature Engineering:** Mean, Standard Deviation, Max, Min  

---

## 📁 Project Structure
```
Earthquake-Prediction/
│
├── earthquake_prediction.py
├── Earthquake_Prediction.ipynb
├── sample_data.csv
├── requirements.txt
└── README.md
```
---

## 🤖 Machine Learning Models Used

- **Linear Regression** – Baseline model  
- **Random Forest Regressor** – Captures complex non-linear patterns  
- **Support Vector Regression (SVR)** – Optimized regression boundary  
- **Stacking** – Ensemble model for improved accuracy  

---

## 📈 Evaluation Metrics

- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Square Error)**  
- **R² Score**  

These metrics measure prediction accuracy and model reliability.

---

## 🚀 How to Run the Project

## 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
## 2️⃣ Run Python Script
 ```bash
python earthquake_prediction.py
```
## OR Run Using Jupyter Notebook
 ```bash
jupyter notebook
```

---

## 🌱 Real-World Impact

Enhances early warning systems

Supports disaster preparedness

Helps reduce economic and human loss

Contributes to resilient infrastructure planning

---

## 🔮 Future Improvements

Implement Deep Learning models (LSTM)

Use real-time streaming seismic data

Improve feature engineering

Combine satellite data with seismic signals

---

## 🔗 References

https://www.kaggle.com/c/LANL-Earthquake-Prediction

https://earthquake.usgs.gov/

Los Alamos National Laboratory Publications

---

## 👨‍💻 Developed By

Tamjid Dhib & Team

Dr. Subhash University

Skill4Program – AI Saksham
