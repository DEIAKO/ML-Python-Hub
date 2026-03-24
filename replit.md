# Restaurant Rating Predictor — Replit Project

## Overview

A Streamlit machine learning web app that predicts restaurant aggregate ratings using the Zomato dataset.

## Architecture

- **Language:** Python 3.11
- **Framework:** Streamlit
- **ML:** scikit-learn RandomForestRegressor
- **Data:** Zomato Restaurants CSV (~9,551 rows) via kagglehub
- **Visualisation:** Plotly

## Key Files

| File | Purpose |
|---|---|
| `ml-app/main.py` | Main Streamlit application (5 tabs) |
| `ml-app/restaurant_data.csv` | Downloaded Zomato dataset |
| `ml-app/.streamlit/config.toml` | Streamlit theme (black/white serif) |
| `ml-app/download_data.py` | Standalone dataset downloader |
| `ml-app/README.md` | Project documentation |

## Environment Secrets

- `KAGGLE_USERNAME` — Kaggle account username
- `KAGGLE_KEY` — Kaggle API key

## Workflow

- **Command:** `streamlit run ml-app/main.py --server.port 5000`
- **Port:** 5000

## Features

- 5-tab Streamlit UI: Home, Data Explorer, Model Performance, Predict, About
- Random Forest with 200 trees, max_depth=12
- Feature importance: Votes (~74%), Cost (~21%), Price range (~5%)
- Batch prediction via CSV upload
- R² ≈ 0.43, RMSE ≈ 0.42, MAE ≈ 0.31

## GitHub

https://github.com/DEIAKO/Restaurant-Recommendation-ML-Project
