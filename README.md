# 50 Startups Multiple Linear Regression

A small interactive web demo for predicting startup profit using multiple linear regression.

## Project goal
This project applies the CRISP-DM workflow to the Kaggle 50 Startups dataset and predicts the `Profit` value using the following inputs:

- R&D Spend
- Administration
- Marketing Spend
- State

## Model
The web demo uses a multiple linear regression model with one-hot encoding for the categorical `State` feature.

## Files

- `index.html` — main page
- `styles.css` — styling
- `app.js` — regression logic and dashboard rendering
- `50_Startups.csv` — dataset
- `vercel.json` — Vercel static deployment config

## Local preview
Open the project in a browser via a local server:

```bash
python -m http.server 8000
```

Then visit:

```text
http://localhost:8000/
```

## Vercel deployment
This project is static and can be deployed directly to Vercel.

1. Push the project to GitHub.
2. Open Vercel.
3. Import the repository.
4. Use the default static site settings.
5. Deploy.

## Metrics displayed
The page shows:

- R-squared (R²)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Notes
The project is intended for learning and demonstration purposes, especially for showing how multiple linear regression works on a real-world business dataset.
