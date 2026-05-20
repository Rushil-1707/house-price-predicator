# 🏠 House Price Prediction

A machine learning web application that predicts California house prices using Random Forest Regression.

## Tech Stack
- **Python** – Core language
- **Scikit-learn** – ML model (Linear Regression + Random Forest)
- **Pandas & NumPy** – Data processing
- **Flask** – Web framework
- **HTML/CSS** – Frontend UI

## Dataset
California Housing Dataset (built into Scikit-learn) — 20,640 samples, 8 features.

## ML Pipeline
1. Load dataset
2. Feature Engineering & Scaling (StandardScaler)
3. Train-Test Split (80/20)
4. Train Linear Regression & Random Forest
5. Evaluate using R² Score and RMSE
6. Save best model (Random Forest) using Pickle

## Model Performance
| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | ~0.60 | ~0.72 |
| Random Forest | ~0.81 | ~0.50 |

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model.py
```

### 3. Run the web app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Project Structure
```
house-price-prediction/
├── model.py              # Train and save ML model
├── app.py                # Flask web application
├── templates/
│   └── index.html        # Web UI
├── requirements.txt      # Dependencies
└── README.md
```

## Author
Rushil Popat | 23IT102 | CSPIT, CHARUSAT
