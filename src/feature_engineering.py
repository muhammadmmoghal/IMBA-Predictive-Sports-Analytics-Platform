# feature_engineering.py
#
# PURPOSE:
#   Transform the clean player-game dataset into a model-ready feature matrix
#   saved to data/processed/features.csv.
#
# PLANNED STEPS:
#   1. Load data/processed/clean_stats.csv.
#   2. Sort by player and game date to ensure correct temporal ordering.
#   3. Compute rolling/lag features per player (no data leakage):
#        - rolling 3-game and 5-game averages for each stat
#        - lag-1 and lag-2 values for key stats
#   4. Add contextual features:
#        - home vs. away indicator
#        - days of rest since last game
#        - opponent strength (e.g., opponent win rate)
#   5. Encode categorical variables (team, opponent, etc.).
#   6. Define the prediction target column (e.g., next-game performance score).
#   7. Drop rows where targets or required features are NaN (start of season).
#   8. Save to data/processed/features.csv.
#
# USAGE (once implemented):
#   python src/feature_engineering.py
