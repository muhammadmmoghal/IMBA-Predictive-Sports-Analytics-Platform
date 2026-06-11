# IMBA Predictive Sports Analytics Platform

An end-to-end sports analytics and machine learning project built with Python.

## Goal

Collect player and game statistics from imbaonline.com, clean the data into a structured player-game dataset, perform exploratory data analysis, train models to predict next-game player performance, and deploy an interactive Streamlit dashboard.

## Project Structure

```
IMBA-Predictive-Sports-Analytics-Platform/
├── data/
│   ├── raw/            # Raw scraped data (unmodified)
│   └── processed/      # Cleaned and feature-engineered data
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── scraper.py          # Scrapes player/game stats from imbaonline.com
│   ├── clean_data.py       # Cleans and standardizes raw data
│   ├── feature_engineering.py  # Builds model-ready features
│   ├── train_model.py      # Trains performance prediction models
│   └── predict.py          # Generates predictions for new games
├── app/
│   └── streamlit_app.py    # Interactive dashboard
├── models/             # Saved trained model artifacts
└── reports/
    └── figures/        # EDA plots and evaluation charts
```

## Tech Stack

- **Data collection:** requests, BeautifulSoup / Selenium
- **Data processing:** Python, Pandas, NumPy, SQL
- **Modeling:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

1. `src/scraper.py` — collect raw data
2. `src/clean_data.py` — clean and normalize
3. `src/feature_engineering.py` — engineer features
4. `src/train_model.py` — train and evaluate models
5. `src/predict.py` — generate predictions
6. `app/streamlit_app.py` — serve the dashboard
