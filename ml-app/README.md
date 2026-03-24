# Restaurant Rating Predictor

A machine learning web application that predicts Zomato restaurant aggregate ratings using a Random Forest Regressor.

## Features

- **Interactive prediction** — Enter cost, price range, and votes to get an instant rating prediction
- **Data Explorer** — Visualise the Zomato dataset with correlation heatmaps and distribution charts
- **Model Performance** — Actual vs Predicted scatter plots, residual histograms, feature importance
- **Batch Prediction** — Upload a CSV and download predictions for multiple restaurants
- **Black & white serif UI** — Clean, editorial aesthetic built with Streamlit

## ML Pipeline

| Step | Detail |
|---|---|
| Dataset | Zomato Restaurants (Kaggle, ~9,551 rows) |
| Usable rows | ~7,403 (rated entries only) |
| Features | Average Cost for two, Price range, Votes |
| Target | Aggregate rating (1–5 scale) |
| Model | RandomForestRegressor (200 trees, max_depth=12) |
| R² | ~0.43 |
| RMSE | ~0.42 |
| MAE | ~0.31 |

## Setup

```bash
pip install streamlit scikit-learn pandas numpy plotly kagglehub
```

Set Kaggle credentials as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Run:

```bash
streamlit run ml-app/main.py --server.port 5000
```

## Repository

https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project
