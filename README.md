# Predictive-Maintenance-MLOps-Pipeline

This project implements an end-to-end MLOps workflow for predictive maintenance using sensor data from industrial machines.
The objective is to predict machine failure types and build a production-ready ML pipeline including validation, training, monitoring, and explainability.

📌 Problem Statement

A manufacturing system collects real-time sensor data such as:
Temperature
Rotational speed
Torque
Tool wear

The goal is to classify whether a machine is:

Operating normally
Or heading toward a specific failure type
🧠 Key Features
✅ Data validation using Pandera
📊 Exploratory Data Analysis (EDA)
⚙️ Feature engineering (Power_W, Temp_diff)
⚖️ Class imbalance handling using SMOTE
🤖 Model training & comparison (4 models)
📈 Experiment tracking using MLflow
🔍 Hyperparameter tuning using Optuna
📉 Drift detection using Evidently
🧾 Model explainability using SHAP
🔄 Retraining decision framework
🏗️ MLOps Pipeline
Validate → Train → Track → Tune → Monitor → Explain → Decide
📂 Project Structure
├── data/
│   ├── train.csv
│   ├── current.csv
│   └── stress.csv
│
├── notebook/
│   └── MLOps_Assignment.ipynb
│
├── artifacts/
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── drift_current.html
│   ├── drift_stress.html
│   └── shap_per_class.png
│
├── requirements.txt
└── README.md
📊 Model Performance
Model	Accuracy	Macro F1
RandomForest	~0.47	~0.35 (Best)
XGBoost	~0.51	~0.33
LightGBM	~0.55	~0.32
LogisticRegression	~0.34	~0.29

👉 Best model: RandomForest (selected using Macro F1)

⚠️ Key Insights
Accuracy is misleading due to class imbalance
Macro F1 provides a better evaluation metric
Rare failure class (TWF) suffers due to data scarcity (~30 samples)
SMOTE improves balance but cannot replace real data
📉 Drift Analysis
Current dataset → stable
Stress dataset → significant drift

Drift observed in:

Rotational speed
Torque
Temperature
Power_W
Temp_diff

👉 Indicates machines operating under higher load conditions

🔍 Explainability (SHAP)
Power_W → drives Power Failure (PWF)
Temp_diff → drives Heat Failure (HDF)
Speed + Torque → drive Overstrain Failure (OSF)

👉 Different failure types have distinct physical causes

🚨 Final Recommendation

Condition: High Power_W or Temp_diff
Risked Failure: PWF / HDF
Action: Implement real-time monitoring and preventive maintenance triggers

🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
XGBoost, LightGBM
MLflow
Optuna
Evidently
SHAP
Pandera
📌 How to Run
pip install -r requirements.txt
jupyter notebook
🎯 Key Learning Outcomes
Handling imbalanced classification problems
Building end-to-end ML pipelines
Applying MLOps concepts in practice
Connecting ML outputs to business decisions
👤 Author

Venkatesh Bammidi
