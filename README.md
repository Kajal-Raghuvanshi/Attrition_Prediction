# Employee Attrition Predictor

A single-file, end-to-end machine learning application that predicts whether an employee is likely to leave the organisation. The backend is powered by **scikit-learn** (Random Forest) and the interactive frontend is built with **Gradio**.

---

## Features

| Tab | Description |
|-----|-------------|
| 📊 **Dashboard** | Five exploratory charts — attrition distribution, age, department, income, and overtime breakdowns |
| 🔮 **Predict** | Fill in 30+ employee attributes and get an instant attrition probability with a visual gauge |
| 📈 **Model Insights** | Feature importance, ROC curve, and confusion matrix for the trained model |

---

## Project Structure

```
IBM_Project/
├── app.py               # All application logic (data, model, UI)
├── Attrition.csv.csv    # IBM HR Analytics dataset
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Prerequisites

- Python 3.9 or later

---

## Installation

```bash
# 1. Clone / download the project
cd IBM_Project

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
python app.py
```

The app starts on **http://localhost:7860** and opens automatically in your default browser.

---

## Dataset

The project uses the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) dataset (`Attrition.csv.csv`).  
Place the file in the same directory as `app.py` before running.

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest |
| Trees | 200 |
| Class weighting | Balanced (handles imbalanced labels) |
| Validation | 5-fold cross-validation (ROC-AUC) |
| Categorical encoding | `LabelEncoder` per feature |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Chart generation |
| `scikit-learn` | Model training and evaluation |
| `gradio` | Interactive web UI |

---

## License

This project is provided for educational and demonstration purposes.
