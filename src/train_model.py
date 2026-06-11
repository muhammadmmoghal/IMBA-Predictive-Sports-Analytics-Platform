# train_model.py
#
# PURPOSE:
#   Train and evaluate machine learning models to predict next-game player
#   performance. Save the best model artifact to models/.
#
# PLANNED STEPS:
#   1. Load data/processed/features.csv.
#   2. Split into train / validation / test sets using a time-based split
#      (no random shuffling — future games must not leak into training).
#   3. Define candidate models (e.g., Ridge Regression, Random Forest,
#      Gradient Boosting).
#   4. Train each model on the training set.
#   5. Evaluate on the validation set using appropriate metrics
#      (MAE, RMSE, R²).
#   6. Select the best model; optionally tune hyperparameters with
#      TimeSeriesSplit cross-validation.
#   7. Retrain the best model on train + validation combined.
#   8. Evaluate final performance on the held-out test set.
#   9. Save the trained model to models/best_model.joblib using joblib.
#   10. Log evaluation metrics to reports/.
#
# USAGE (once implemented):
#   python src/train_model.py
